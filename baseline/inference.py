"""
Inference script for bone fracture reduction.

Usage:
    python inference.py                                 # default
    python inference.py --config configs/test_pose.yaml # pose mode
    python inference.py --debug_vis --max_iters 3
"""

import os
import json
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf

from models.AssemblyNet.transformer import AssemblyTransformer
from models.AssemblyNet.lightning_module import AssemblyLightningModule
from models.AssemblyNet.utils import quaternion_to_matrix
from datasets.fractures.eval import load_obj_fragments, sub_sample
from datasets.transform import recenter_pc, rescale_pc

# ============================================================
# Default config - all settings can be overridden via CLI
# ============================================================
CONFIG_PATH = "configs/test.yaml"
# ============================================================

def parse_args():
    """Parse command-line arguments (all optional, defaults come from the YAML config)."""
    import argparse
    parser = argparse.ArgumentParser(description="Bone fracture reduction inference")
    parser.add_argument("--config", type=str, default=None,
                        help=f"Config YAML path (default: {CONFIG_PATH})")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Override input_dir from config")
    parser.add_argument("--npoints", type=int, default=None,
                        help="Override npoints from config")
    parser.add_argument("--max_iters", type=int, default=None,
                        help="Override max_iters from config")
    parser.add_argument("--debug_vis", action="store_true",
                        help="Enable trimesh debug visualizations")
    return parser.parse_args()


def load_config_and_model(config_path, device, debug_vis=None):
    """
    Load test config, build model, and load weights.

    Test config provides: input_dir, npoints, max_iters, checkpoints, experiment_name
    Model config (model_config path) provides: output_type, transformer_model

    Returns:
        model: AssemblyLightningModule in eval mode
        cfg: merged OmegaConf dict
    """
    test_cfg = OmegaConf.load(config_path)

    # Load model-specific config (has output_type and transformer_model)
    model_cfg_path = Path(config_path).parent / f"{test_cfg.model_config}.yaml"
    model_cfg = OmegaConf.load(model_cfg_path)

    # Merge: test config values override model config values
    cfg = OmegaConf.merge(model_cfg, test_cfg)

    output_type = cfg.output_type
    tm = cfg.transformer_model
    experiment_name = cfg.experiment_name

    transformer = AssemblyTransformer(
        embed_dim=tm.embed_dim,
        num_layers=tm.num_layers,
        num_heads=tm.num_heads,
        dropout_rate=tm.dropout_rate,
        max_parts=tm.max_parts,
        output_type=output_type,
    )

    # debug_vis from CLI override, else from config
    vis = cfg.get("debug_vis", False)

    training_mode = cfg.get("training_mode", "full")
    lora_rank = cfg.get("lora_rank", 16)
    lora_alpha = cfg.get("lora_alpha", 32.0)
    model = AssemblyLightningModule(
        transformer_model=transformer,
        optimizer=partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.01),
        output_type=output_type,
        training_mode=training_mode,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        debug_vis=vis,
    )

    # Resolve checkpoints: explicit path or auto-discover latest epoch
    if cfg.get("checkpoint"):
        checkpoint_path = str(Path(cfg.checkpoint))
    else:
        exp_dir = Path("output") / experiment_name
        checkpoints = sorted(exp_dir.glob("epoch-epoch=*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {exp_dir}")
        checkpoint_path = str(checkpoints[-1])

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    return model, cfg


def discover_samples(input_dir):
    """
    Find all patient sample directories containing OBJ files.
    Expected structure: input_dir/sample_name/xxx.obj
    Returns: List of (sample_name, sample_dir, obj_path) tuples, sorted by name.
    """
    samples = []
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    for sample_dir in sorted(input_path.iterdir()):
        if not sample_dir.is_dir():
            continue
        obj_files = list(sample_dir.glob("*.obj"))
        if len(obj_files) == 0:
            print(f"  [WARN] No OBJ file in {sample_dir.name}, skipping.")
            continue
        if len(obj_files) > 1:
            print(f"  [WARN] Multiple OBJ files in {sample_dir.name}, using: {obj_files[0].name}")
        samples.append((sample_dir.name, str(sample_dir), str(obj_files[0])))
    return samples


def build_Tinit(centroid, scale):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -centroid / scale
    T[:3, :3] = np.diag(1.0 / np.array([scale, scale, scale], dtype=np.float32))
    return T


def transform_pose_to_original_space(T_pred, centroid, scale):
    T_init = build_Tinit(centroid, scale)
    T_init_inv = np.linalg.inv(T_init)
    T_final = T_init_inv @ T_pred @ T_init
    return T_final


def apply_transform_to_fragment_coords(fragment_coords, T):
    ones = np.ones((fragment_coords.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([fragment_coords, ones], axis=1)
    transformed = (T @ pts_h.T).T
    return transformed[:, :3]


def apply_rotation_to_fragment_normals(fragment_normals, T):
    R = T[:3, :3]
    return (R @ fragment_normals.T).T


def build_fragment_data_from_meshdict(meshdict, npoints):
    sampled = sub_sample(meshdict, npoints=npoints, min_points=20)
    fragment_data = []
    for bone_type in ["SA", "LI", "RI"]:
        fragments = sampled.get(bone_type, {})
        for frag_name in sorted(fragments.keys()):
            coords = fragments[frag_name]["coords"]
            normals = fragments[frag_name]["normals"]
            fragment_data.append((coords, normals))
    return fragment_data


def run_inference_pass(model, all_points, all_normals, points_per_part, device, max_parts):
    points_per_part_padded = np.zeros(max_parts, dtype=np.int32)
    points_per_part_padded[:len(points_per_part)] = points_per_part

    tensor_dict = {
        "pointclouds": torch.from_numpy(all_points).to(device),
        "pointclouds_normals": torch.from_numpy(all_normals).to(device),
        "points_per_part": torch.from_numpy(points_per_part_padded).unsqueeze(0).to(device),
    }

    use_amp = (device == "cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        output = model(tensor_dict)

    # Free input tensors from GPU immediately — only need quat/trans from output
    del tensor_dict
    torch.cuda.empty_cache()

    return output


def run_iterative_inference(model, fragment_data, device, output_type, max_parts,
                            max_iters=5, convergence_threshold=1e-4):
    num_parts = len(fragment_data)
    cumulative_transforms = [np.eye(4, dtype=np.float32) for _ in range(num_parts)]
    iter_deltas = []

    current_coords = [fd[0].copy() for fd in fragment_data]
    current_normals = [fd[1].copy() for fd in fragment_data]

    print(f"  Starting iterative inference ({max_iters} max iters, mode={output_type})...")

    for iter_idx in range(max_iters):
        all_points = np.concatenate(current_coords, axis=0).astype(np.float32)
        all_points, centroid = recenter_pc(all_points)
        all_points, scale = rescale_pc(all_points)

        offsets = [0]
        for fc in current_coords:
            offsets.append(offsets[-1] + len(fc))
        norm_coords = [all_points[offsets[i]:offsets[i+1]] for i in range(num_parts)]

        all_normals = np.concatenate(current_normals, axis=0).astype(np.float32)
        points_per_part = np.array([len(nc) for nc in norm_coords], dtype=np.int32)

        output = run_inference_pass(model, all_points, all_normals, points_per_part, device, max_parts)

        # Immediately move to CPU and delete GPU tensors
        quat = output["quat"].float().detach().cpu()
        trans = output["trans"].float().detach().cpu()
        rot = quaternion_to_matrix(quat)
        del output
        torch.cuda.empty_cache()

        T_preds = []
        for i in range(num_parts):
            T_pred = np.eye(4, dtype=np.float32)
            T_pred[:3, :3] = rot[i].numpy()
            T_pred[:3, 3] = trans[i].numpy()
            T_preds.append(T_pred)

        T_is = [transform_pose_to_original_space(T_preds[i], centroid, scale) for i in range(num_parts)]

        max_delta = 0.0
        for i in range(num_parts):
            cumulative_transforms[i] = T_is[i] @ cumulative_transforms[i]
            current_coords[i] = apply_transform_to_fragment_coords(current_coords[i], T_is[i])
            current_normals[i] = apply_rotation_to_fragment_normals(current_normals[i], T_is[i])
            delta = np.linalg.norm(T_is[i][:3, 3])
            max_delta = max(max_delta, delta)

        iter_deltas.append(max_delta)
        print(f"  Iter {iter_idx + 1}: max_trans_delta={max_delta:.6f}")

        if max_delta < convergence_threshold:
            print(f"  Converged at iteration {iter_idx + 1}!")
            return cumulative_transforms, iter_idx + 1, True, iter_deltas

    print(f"  Reached max iterations ({max_iters}), not converged.")
    return cumulative_transforms, max_iters, False, iter_deltas


def normalize_transforms_by_first_sa(cumulative_transforms, meshdict):
    """
    Re-express all transforms relative to the first SA fragment.
    The first SA fragment gets identity (stays in place); others are relative to it.
    """
    if not meshdict.get("SA"):
        return cumulative_transforms
    # SA is always iterated first, so index 0 is the first SA fragment
    T_sa0_inv = np.linalg.inv(cumulative_transforms[0])
    return [T_sa0_inv @ T for T in cumulative_transforms]


def build_final_results(cumulative_transforms, fragment_names):
    results = []
    for i, frag_name in enumerate(fragment_names):
        results.append({
            "fragment_id": frag_name,
            "transformation": cumulative_transforms[i].tolist(),
        })
    return results


def _scale_toward_identity(results, alpha, gate_rot=0.0, gate_trans=0.0, anchor_id="1"):
    """[v4.1/v4.2] Overshoot-corrected, displacement-GATED blend of the predicted poses.

    The model (trained mainly on large-displacement simulated fractures) OVERSHOOTS the
    near-assembled clinical fragments. Two corrections, both validated on 170 clinical cases
    with the official evaluator:
      * [v4.1] scale each transform toward identity by `alpha` (rot angle*alpha, trans*alpha).
      * [v4.2] displacement GATE: a fragment is MOVED only when the model itself predicts a
        large displacement (raw pred rot >= gate_rot deg OR raw pred trans >= gate_trans mm);
        otherwise it is left at IDENTITY. Rescues the badly-displaced fragments while never
        disturbing the ones already in place -> preserves Part-Accuracy (the tight-threshold
        metric that a blanket blend degrades). alpha=1 & gate=0 => plain v4.0 behaviour.
    Numpy-only (Rodrigues), no scipy dependency. The anchor fragment stays identity."""
    import numpy as _np
    I = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
    out = []
    for e in results:
        T = _np.asarray(e["transformation"], dtype=float)
        R = T[:3, :3]
        ang = float(_np.arccos(_np.clip((_np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))  # rad
        tmag = float(_np.linalg.norm(T[:3, 3]))
        gated_out = (
            str(e["fragment_id"]) == str(anchor_id)
            or ((gate_rot > 0.0 or gate_trans > 0.0)
                and _np.degrees(ang) < gate_rot and tmag < gate_trans)
        )
        if gated_out:
            out.append({"fragment_id": e["fragment_id"], "transformation": I})
            continue
        s = _np.sin(ang)
        if abs(s) < 1e-8:
            Rs = _np.eye(3) if ang < 1.5 else R  # ~0 -> identity; ~pi -> leave as-is (rare)
        else:
            axis = _np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * s)
            a2 = ang * alpha
            K = _np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            Rs = _np.eye(3) + _np.sin(a2) * K + (1.0 - _np.cos(a2)) * (K @ K)
        M = _np.eye(4)
        M[:3, :3] = Rs
        M[:3, 3] = T[:3, 3] * alpha
        out.append({"fragment_id": e["fragment_id"], "transformation": M.tolist()})
    return out


def save_prediction(sample_dir, results, output_type, training_mode="full", result_suffix=""):
    if result_suffix:
        suffix = f"_{result_suffix}"
    elif output_type != "coords":
        suffix = f"_{output_type}"
    elif training_mode != "full":
        suffix = f"_{training_mode}"
    else:
        suffix = ""
    filename = f"reduction-poses-matrices{suffix}.json"
    json_path = os.path.join(sample_dir, filename)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    return json_path


def main():
    args = parse_args()

    config_path = args.config if args.config is not None else CONFIG_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading config: {config_path}")
    model, cfg = load_config_and_model(config_path, device, debug_vis=args.debug_vis)
    output_type = cfg.output_type
    training_mode = cfg.get("training_mode", "full")
    max_parts = cfg.transformer_model.max_parts

    input_dir = args.input_dir if args.input_dir is not None else cfg.input_dir
    npoints = args.npoints if args.npoints is not None else cfg.get("npoints", 5000)
    max_iters = args.max_iters if args.max_iters is not None else cfg.get("max_iters", 5)
    convergence_threshold = cfg.get("convergence_threshold", 1e-4)
    debug_vis = args.debug_vis if args.debug_vis else cfg.get("debug_vis", False)

    print(f"  output_type={output_type}, training_mode={training_mode}, max_parts={max_parts}")
    print(f"  debug_vis={debug_vis}, max_iters={max_iters}, npoints={npoints}")
    print(f"  Model loaded.\n")

    print(f"Scanning: {input_dir}")
    samples = discover_samples(input_dir)
    print(f"Found {len(samples)} sample(s).\n")

    if not samples:
        print("No samples found.")
        return

    num_success = 0
    num_failed = 0

    for idx, (sample_name, sample_dir, obj_path) in enumerate(samples):
        print(f"\n[{idx + 1}/{len(samples)}] {sample_name}")
        t_start = time.time()

        try:
            meshdict = load_obj_fragments(obj_path, verbose=False)
            fragment_data = build_fragment_data_from_meshdict(meshdict, npoints=npoints)
            num_parts = len(fragment_data)
            print(f"  Fragments: {num_parts}")

            if debug_vis:
                print("  Visualizing input fragments...")
                from datasets.vis_func import visualize_fragments; visualize_fragments(meshdict)

            cumulative_transforms, iter_used, converged, iter_deltas = run_iterative_inference(
                model, fragment_data, device,
                output_type=output_type,
                max_parts=max_parts,
                max_iters=max_iters,
                convergence_threshold=convergence_threshold,
            )

            fragment_names = []
            for bone_type in ["SA", "LI", "RI"]:
                fragments = meshdict.get(bone_type, {})
                for frag_name in sorted(fragments.keys()):
                    fragment_names.append(frag_name)

            normalized_transforms = normalize_transforms_by_first_sa(cumulative_transforms, meshdict)
            results = build_final_results(normalized_transforms, fragment_names)
            pred_scale = float(cfg.get("pred_scale", 1.0))       # [v4.1] overshoot scale (0.6)
            gate_rot = float(cfg.get("pred_gate_rot", 0.0))       # [v4.2] deg; move only if >=
            gate_trans = float(cfg.get("pred_gate_trans", 0.0))   # [v4.2] mm;  move only if >=
            if pred_scale != 1.0 or gate_rot > 0.0 or gate_trans > 0.0:
                results = _scale_toward_identity(results, pred_scale, gate_rot, gate_trans)
            result_suffix = cfg.get("result_suffix", "")
            json_path = save_prediction(sample_dir, results, output_type, training_mode, result_suffix)

            if debug_vis:
                print("  Visualizing predicted assembly...")
                from datasets.vis_func import visualize_transformed_fragments; visualize_transformed_fragments(meshdict, results)

            elapsed = time.time() - t_start
            num_success += 1

            delta_str = ", ".join([f"{d:.6f}" for d in iter_deltas])
            print(f"  Iters: {iter_used}/{max_iters}, converged={converged}")
            print(f"  Deltas: [{delta_str}]")
            print(f"  Saved: {json_path}  ({elapsed:.2f}s)")

        except Exception as e:
            import traceback
            elapsed = time.time() - t_start
            num_failed += 1
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"Total: {len(samples)}  Success: {num_success}  Failed: {num_failed}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()


