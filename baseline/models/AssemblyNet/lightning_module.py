"""Pure Transformer Lightning Module for Point Cloud Assembly."""

from __future__ import annotations

import os
from functools import partial
from typing import Optional
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

class AssemblyLightningModule(L.LightningModule):
    def __init__(
            self,
            transformer_model: nn.Module,
            optimizer: "partial[torch.optim.Optimizer]",
            output_type: str = "coords",
            training_mode: str = "full",
            lora_rank: int = 16,
            lora_alpha: float = 32.0,
            save_vis_path: str = "./output",
            debug_vis: bool = False,
            checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.transformer_model = transformer_model
        self.optimizer = optimizer
        self.output_type = output_type
        self.training_mode = training_mode
        self.debug_vis = debug_vis

        self.base_vis_path = save_vis_path
        self.save_vis_path_val = None

        if checkpoint is not None:
            self._load_pretrained_weights(checkpoint)

        if training_mode == "lora":
            from .lora import apply_lora_to_transformer
            apply_lora_to_transformer(self.transformer_model, rank=lora_rank, alpha=lora_alpha)

    def on_fit_start(self):
        if self.trainer.is_global_zero:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            mode_str = f"mode={self.output_type}  training={self.training_mode}"
            param_str = f"params={trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total"
            print(f"[Model] {mode_str}  {param_str}")

    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained weights into the transformer before LoRA wrapping."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        prefix = "transformer_model."
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                stripped[k[len(prefix):]] = v
            else:
                stripped[k] = v

        missing, unexpected = self.transformer_model.load_state_dict(stripped, strict=False)

        if os.environ.get("LOCAL_RANK", "0") == "0":
            loaded = len(stripped) - len(missing)
            print(f"[Pretrained] Loaded {loaded} keys from {checkpoint_path}")
            if missing:
                print(f"[Pretrained]   Missing keys: {missing}")
            if unexpected:
                print(f"[Pretrained]   Unexpected keys: {unexpected}")

    def setup(self, stage: str):
        """Set up output directories for validation/testing stages."""
        if stage in ['fit', 'validate']:
            self.save_vis_path_val = os.path.join(self.base_vis_path, "valsets_results")
            os.makedirs(self.save_vis_path_val, exist_ok=True)

    def on_train_epoch_start(self):
        """Dynamic reshuffling at the start of each epoch to increase data variety."""
        super().on_train_epoch_start()

        train_loader = self.trainer.train_dataloader
        if train_loader is not None:
            dataset = train_loader.dataset
            if hasattr(dataset, 'datasets'):
                for ds in dataset.datasets:
                    if getattr(ds, 'split', None) == 'train' and hasattr(ds, 'reshuffle'):
                        ds.reshuffle()
            else:
                if getattr(dataset, 'split', None) == 'train' and hasattr(dataset, 'reshuffle'):
                    dataset.reshuffle()

    def forward(self, data_dict: dict):
        """Single-step forward pass through the transformer."""
        points_per_part = data_dict["points_per_part"]
        input_coords = data_dict["pointclouds"]
        input_normals = data_dict["pointclouds_normals"]

        output_dict = self.transformer_model(
            input_coords=input_coords,
            input_normals=input_normals,
            points_per_part=points_per_part,
        )

        if "part_batch_ids" not in output_dict:
            part_valids = points_per_part != 0
            seq_len = points_per_part[part_valids]
            output_dict["part_batch_ids"] = torch.repeat_interleave(
                torch.arange(len(seq_len), device=self.device), seq_len
            )

        if self.output_type == "pose":
            if self.debug_vis:
                pred_assemble = apply_pose_to_points(
                    input_coords,
                    output_dict["trans"],
                    output_dict["quat"],
                    output_dict["part_batch_ids"]
                )
                debug_visualize_pred_assemble(input_coords, output_dict["part_batch_ids"])
                debug_visualize_coords_vs_pred(input_coords, pred_assemble, output_dict["part_batch_ids"])
        elif self.output_type == "coords":
            if not self.training or self.debug_vis:
                quat, trans = extract_poses_from_coords(
                    input_coords,
                    output_dict["pred_coords"],
                    output_dict["part_batch_ids"]
                )
                output_dict["quat"] = quat
                output_dict["trans"] = trans
            if self.debug_vis:
                debug_visualize_coords_vs_pred(
                    input_coords,
                    output_dict["pred_coords"],
                    output_dict["part_batch_ids"]
                )
        return output_dict

    def loss(self, output_dict: dict, data_dict: dict):
        """Compute the regression loss based on the output type."""
        gt_coords = data_dict["pointclouds_gt"]

        if self.output_type == "coords":
            pred_coords = output_dict["pred_coords"]
            loss = F.mse_loss(pred_coords, gt_coords, reduction="mean")
            return {"loss": loss}

        elif self.output_type == "pose":
            pred_quat, pred_trans = output_dict["quat"], output_dict["trans"]

            pred_coords = apply_pose_to_points(
                data_dict["pointclouds"],
                pred_trans,
                pred_quat,
                output_dict["part_batch_ids"]
            )
            loss = F.mse_loss(pred_coords, gt_coords, reduction="mean")
            return {"loss": loss}

    def training_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict, data_dict)

        loss_val = loss_dict["loss"]
        batch_size = data_dict["points_per_part"].shape[0]

        # Sync NaN state across ALL ranks — if ANY rank has non-finite loss, ALL skip
        is_finite_int = torch.isfinite(loss_val).int()
        if self.trainer.world_size > 1:
            torch.distributed.all_reduce(is_finite_int, op=torch.distributed.ReduceOp.MIN)
        is_finite = is_finite_int.bool()

        if not is_finite:
            print(f"[Rank {self.global_rank}] Non-finite loss at step {self.global_step}, all ranks skipping.")
            self.log("train/loss", float('nan'), on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True, batch_size=batch_size)
            return loss_val * 0.0

        self.log("train/loss", loss_val, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch_size)

        if "loss_rot" in loss_dict:
            self.log("train/loss_rot", loss_dict["loss_rot"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)
        if "loss_trans" in loss_dict:
            self.log("train/loss_trans", loss_dict["loss_trans"], on_step=True, on_epoch=False, sync_dist=True,
                     batch_size=batch_size)

        return loss_val

    def validation_step(self, data_dict: dict, batch_idx: int, dataloader_idx: int = 0):
        output_dict = self.forward(data_dict)
        loss_dict = self.loss(output_dict, data_dict)
        batch_size = data_dict["points_per_part"].shape[0]

        self.log("val/loss", loss_dict["loss"], on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        if "quat" in output_dict and "trans" in output_dict:
            valid_mask = data_dict["points_per_part"] > 0
            gt_quat = data_dict["quat_gt"][valid_mask]
            gt_trans = data_dict["trans_gt"][valid_mask]
            pred_quat = output_dict["quat"]
            pred_trans = output_dict["trans"]

            with torch.autocast(device_type=self.device.type, enabled=False):
                rot_err = rot_geodesic_deg(pred_quat.float(), gt_quat.float())
                trans_err = torch.norm(pred_trans.float() - gt_trans.float(), dim=-1)

                if "pred_coords" in output_dict:
                    pred_assembled = output_dict["pred_coords"].float()
                else:
                    pred_assembled = apply_pose_to_points(
                        data_dict["pointclouds"].float(),
                        pred_trans.float(), pred_quat.float(),
                        output_dict["part_batch_ids"],
                    )
                tre = torch.sqrt(F.mse_loss(
                    pred_assembled, data_dict["pointclouds_gt"].float(), reduction="none"
                ).mean(dim=-1)).mean()

            n_parts = rot_err.shape[0]
            self.log("val/rot_err_deg", rot_err.mean(), on_epoch=True, sync_dist=True, batch_size=n_parts)
            self.log("val/trans_err", trans_err.mean(), on_epoch=True, sync_dist=True, batch_size=n_parts)
            self.log("val/tre", tre, on_epoch=True, sync_dist=True, batch_size=batch_size)

        if self.debug_vis and batch_idx < 10:
            if self.output_type == "coords":
                pred_assemble = enforce_rigid_transform_svd(
                    src_points=data_dict["pointclouds"],
                    pred_points=output_dict["pred_coords"],
                    part_batch_ids=output_dict["part_batch_ids"]
                )
            else:
                pred_assemble = apply_pose_to_points(
                    data_dict["pointclouds"],
                    output_dict["trans"],
                    output_dict["quat"],
                    output_dict["part_batch_ids"]
                )

            output_results(
                pred_assemble=pred_assemble,
                data_dict=data_dict,
                output_dict=output_dict,
                save_path=self.save_vis_path_val,
                save_all=False,
                save_gt=True
            )

        return loss_dict["loss"]

    def configure_optimizers(self):
        """Standard optimizer configuration with OneCycleLR scheduler."""
        # Exclude biases and 1D params (LayerNorm, etc.) from weight decay
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim <= 1 or name.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)

        base_lr = self.optimizer.keywords.get("lr", 1e-4)
        wd = self.optimizer.keywords.get("weight_decay", 0.01)
        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        optimizer = self.optimizer(param_groups)

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=total_steps,
            pct_start=0.05,
            anneal_strategy='cos',
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}
        }
