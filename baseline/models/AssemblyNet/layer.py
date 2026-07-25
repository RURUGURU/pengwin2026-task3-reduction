"""Transformer layer for Point Cloud Assembly."""

try:
    import flash_attn_interface as flash_attn
    flash_version = 3
except ImportError:
    try:
        import flash_attn
        flash_version = 2
    except ImportError:
        flash_attn = None
        flash_version = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import MultiHeadRMSNorm


class _GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, out_dim: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            _GEGLU(),
            nn.Linear(dim * mult, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """Transformer layer with Intra- and Inter-Fragment Attention."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        qkv_proj_bias: bool = False,
        attn_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.dropout = dropout
        self.attn_dtype = attn_dtype

        self.intra_prenorm = nn.LayerNorm(dim)
        self.intra_qkv_proj = nn.Linear(dim, 3 * self.inner_dim, bias=qkv_proj_bias)
        self.intra_q_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.intra_k_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.intra_out_proj = nn.Linear(self.inner_dim, dim, bias=qkv_proj_bias)

        self.inter_prenorm = nn.LayerNorm(dim)
        self.inter_qkv_proj = nn.Linear(dim, 3 * self.inner_dim, bias=qkv_proj_bias)
        self.inter_q_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.inter_k_norm = MultiHeadRMSNorm(self.attention_head_dim, num_attention_heads)
        self.inter_out_proj = nn.Linear(self.inner_dim, dim, bias=qkv_proj_bias)

        self.ff_norm = nn.LayerNorm(dim)
        self.ff = _FeedForward(dim, dim, mult=4)

    def _run_flash_attn_varlen(self, q, k, v, cu_seqlens, max_seqlen, dtype):
        if flash_attn is None:
            return self._run_attention_standard(q, k, v, cu_seqlens, max_seqlen, dtype)

        if flash_version == 3:
            attn_output = flash_attn.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                softmax_scale=None, causal=False,
            )[0]
        else:
            attn_output = flash_attn.flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=None, causal=False,
            )
        return attn_output.to(dtype)

    def _run_attention_standard(self, q, k, v, cu_seqlens, max_seqlen, dtype):
        """Memory-efficient varlen attention WITHOUT flash-attn (faithful to flash_attn_varlen_func).

        The original fallback padded every segment to max_seqlen and passed a FLOAT attn_mask, which
        forces PyTorch's SDPA MATH backend to materialize a [B, H, S, S] score matrix -> OOM at batch>1
        on real point clouds (S up to ~15k). Instead we run SDPA per varlen segment (defined by
        cu_seqlens) with NO padding and NO mask, so SDPA picks its mem-efficient / flash backend and
        never materializes S^2. Same block-diagonal result as flash_attn_varlen_func; fits memory ->
        enables batch>1 (no flash-attn / no sm_120 wheel needed)."""
        result = q.new_empty(q.shape[0], q.shape[1], q.shape[2])  # [N, H, D], N = total points
        cu = cu_seqlens.tolist()
        dp = self.dropout if self.training else 0.0
        for i in range(len(cu) - 1):
            s, e = int(cu[i]), int(cu[i + 1])
            if e <= s:
                continue
            qi = q[s:e].transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
            ki = k[s:e].transpose(0, 1).unsqueeze(0)
            vi = v[s:e].transpose(0, 1).unsqueeze(0)
            oi = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dp)  # no mask -> mem-efficient
            result[s:e] = oi.squeeze(0).transpose(0, 1)  # [L, H, D]
        return result.to(dtype)

    def _run_attention(self, hidden_states, prenorm_fn, qkv_proj_fn, q_norm, k_norm, out_proj_fn, cu_seqlens, max_seqlen):
        n_points = hidden_states.shape[0]
        dtype = hidden_states.dtype

        x = prenorm_fn(hidden_states)
        qkv = qkv_proj_fn(x).reshape(n_points, 3, self.num_attention_heads, self.attention_head_dim)
        q, k, v = qkv.unbind(dim=1)

        q = q_norm(q).to(v.dtype)
        k = k_norm(k).to(v.dtype)

        attn_output = self._run_flash_attn_varlen(q, k, v, cu_seqlens, max_seqlen, dtype)
        return hidden_states + out_proj_fn(attn_output.flatten(1))

    def forward(self, hidden_states, intra_attn_cu_seqlens, intra_attn_max_seqlen,
                inter_attn_cu_seqlens, inter_attn_max_seqlen, batch=None):
        hidden_states = self._run_attention(
            hidden_states, self.intra_prenorm, self.intra_qkv_proj,
            self.intra_q_norm, self.intra_k_norm, self.intra_out_proj,
            intra_attn_cu_seqlens, intra_attn_max_seqlen,
        )
        hidden_states = self._run_attention(
            hidden_states, self.inter_prenorm, self.inter_qkv_proj,
            self.inter_q_norm, self.inter_k_norm, self.inter_out_proj,
            inter_attn_cu_seqlens, inter_attn_max_seqlen,
        )
        x = self.ff_norm(hidden_states)
        return hidden_states + self.ff(x)
