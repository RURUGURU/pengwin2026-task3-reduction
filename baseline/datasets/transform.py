import random

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as SciPyRot
from scipy.stats import truncnorm

def recenter_pc(pc):
    """Center point cloud by computing bounding box center."""
    centroid = (pc.max(axis=0) + pc.min(axis=0)) / 2
    return pc - centroid[None], centroid[None]


def rotate_pc(pc, normal=None, numpy_rng=None):
    """Apply random rotation to point cloud."""
    if numpy_rng is None:
        numpy_rng = np.random

    rot_mat = R.random(random_state=numpy_rng).as_matrix()
    rotated_pc = (rot_mat @ pc.T).T
    quat_gt = R.from_matrix(rot_mat.T).as_quat()
    quat_gt = quat_gt[[3, 0, 1, 2]]
    if normal is None:
        return rotated_pc, None, quat_gt, rot_mat

    rotated_normal = (rot_mat @ normal.T).T
    return rotated_pc, rotated_normal, quat_gt, rot_mat


def flip_x_axis(pc, normals=None):
    """Flip point cloud along x-axis (mirror transformation)."""
    flip_matrix = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    flipped_pc = pc @ flip_matrix.T
    quat_flip = np.array([0., 1., 0., 0.])

    if normals is None:
        return flipped_pc, None, quat_flip

    normals_flipped = normals @ flip_matrix.T
    return flipped_pc, normals_flipped, quat_flip


def rescale_pc(pc):
    """Rescale point cloud to unit sphere."""
    distances = np.linalg.norm(pc, axis=1)
    scale = np.max(distances)
    pc /= scale
    return pc, scale


def apply_clinical_pose_variation(pointcloud, max_angle_deg=15, max_translation_mm=15, seed=None, np_rng=None):
    """
    Apply random rotation around fragment center + random translation.
    Returns transformed point cloud and equivalent (R, t_eq).
    """
    if np_rng is None:
        np_rng = np.random

    center = np.mean(pointcloud, axis=0)

    def truncated_normal(mean, std, clip):
        a, b = -clip/std, clip/std
        return truncnorm.rvs(a, b, loc=mean, scale=std, random_state=np_rng)
    
    #angles = np.deg2rad(np_rng.uniform(-max_angle_deg, max_angle_deg, size=3))
    angles = np.deg2rad([truncated_normal(0, max_angle_deg/3 , max_angle_deg) for _ in range(3)])
    rot = R.from_euler('xyz', angles)
    rot_mat = rot.as_matrix()
    quat = rot.as_quat()

    translation_rand = np.array([truncated_normal(0, max_translation_mm/3, max_translation_mm) for _ in range(3)])
    #translation_rand = np_rng.uniform(-max_translation_mm, max_translation_mm, size=3)

    pc_centered = pointcloud - center
    pc_rot = (rot_mat @ pc_centered.T).T
    pc_rot_back = pc_rot + center
    pointcloud_transformed_seq = pc_rot_back + translation_rand

    translation_eq = -rot_mat @ center + center + translation_rand

    return pointcloud_transformed_seq, quat, translation_eq, rot_mat


def process_fragments_clinical_pose(pointclouds_gt, pointclouds_normals_gt, point_bone_ids, offset, np_rng=None):
    """
    Apply clinical pose variation to fragments:
    1. Apply first fragment's pose to entire GT
    2. Compute recovery transforms for each fragment
    3. Apply centering and scaling globally
    """
    if np_rng is None:
        np_rng = np.random

    num_parts = len(offset) - 1

    absolute_poses = []
    for part_idx in range(num_parts):
        start = offset[part_idx]
        end = offset[part_idx + 1]
        pc = pointclouds_gt[start:end]

        pc_posed, q_abs, t_abs, rot_abs = apply_clinical_pose_variation(pc, np_rng=np_rng)
        absolute_poses.append({
            'rot_mat': rot_abs,
            't': t_abs
        })

    R0 = absolute_poses[0]['rot_mat']
    T0 = absolute_poses[0]['t']

    pointclouds_gt = pointclouds_gt @ R0.T + T0
    pointclouds_normals_gt = pointclouds_normals_gt @ R0.T

    pointclouds, pointclouds_normals = [], []
    forward_transforms = []

    for part_idx in range(num_parts):
        start = offset[part_idx]
        end = offset[part_idx + 1]

        pc_new_gt = pointclouds_gt[start:end]
        nm_new_gt = pointclouds_normals_gt[start:end]

        Ri = absolute_poses[part_idx]['rot_mat']
        Ti = absolute_poses[part_idx]['t']

        R_rel = Ri @ R0.T
        T_rel = Ti - (T0 @ R_rel.T)

        forward_transforms.append((R_rel, T_rel))

        pc_posed = pc_new_gt @ R_rel.T + T_rel
        nm_posed = nm_new_gt @ R_rel.T

        order = np_rng.permutation(len(pc_posed))

        pc_posed = pc_posed[order]
        nm_posed = nm_posed[order]
        pointclouds_gt[start:end] = pointclouds_gt[start:end][order]
        pointclouds_normals_gt[start:end] = pointclouds_normals_gt[start:end][order]
        point_bone_ids[start:end] = point_bone_ids[start:end][order]

        pointclouds.append(pc_posed)
        pointclouds_normals.append(nm_posed)

    pointclouds = np.concatenate(pointclouds).astype(np.float32)
    pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)

    pointclouds, translation_final = recenter_pc(pointclouds)
    pointclouds, scales = rescale_pc(pointclouds)
    pointclouds_gt = (pointclouds_gt - translation_final) / scales

    quaternions, translations_rec = [], []

    for R_rel, T_rel in forward_transforms:
        R_rec = R_rel.T
        T_rec = ((translation_final - T_rel) @ R_rel - translation_final) / scales

        q_xyzw = SciPyRot.from_matrix(R_rec).as_quat()
        q_rec = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        quaternions.append(q_rec)
        translations_rec.append(T_rec.flatten().astype(np.float32))

    quaternions = np.stack(quaternions).astype(np.float32)
    translations = np.stack(translations_rec).astype(np.float32)

    return pointclouds_gt, pointclouds_normals_gt, point_bone_ids, pointclouds, pointclouds_normals, quaternions, translations


def process_fragments_with_given_poses(
    pointclouds_gt, pointclouds_normals_gt, point_bone_ids, offset,
    per_frag_rot, per_frag_trans, np_rng=None,
):
    """Same as process_fragments_clinical_pose but uses given per-fragment poses
    instead of random ones.  Used to reconstruct real clinical fracture configurations.

    Args:
        per_frag_rot: (M, 3, 3) rotation matrices assembled→scattered per fragment.
        per_frag_trans: (M, 3) translation vectors assembled→scattered per fragment.
    """
    if np_rng is None:
        np_rng = np.random

    num_parts = len(offset) - 1

    # Build absolute poses from provided transforms
    absolute_poses = [
        {"rot_mat": per_frag_rot[i], "t": per_frag_trans[i]}
        for i in range(num_parts)
    ]

    # --- remainder is identical to process_fragments_clinical_pose ---
    R0 = absolute_poses[0]["rot_mat"]
    T0 = absolute_poses[0]["t"]

    pointclouds_gt = pointclouds_gt @ R0.T + T0
    pointclouds_normals_gt = pointclouds_normals_gt @ R0.T

    pointclouds, pointclouds_normals = [], []
    forward_transforms = []

    for part_idx in range(num_parts):
        start = offset[part_idx]
        end = offset[part_idx + 1]

        pc_new_gt = pointclouds_gt[start:end]
        nm_new_gt = pointclouds_normals_gt[start:end]

        Ri = absolute_poses[part_idx]["rot_mat"]
        Ti = absolute_poses[part_idx]["t"]

        R_rel = Ri @ R0.T
        T_rel = Ti - (T0 @ R_rel.T)

        forward_transforms.append((R_rel, T_rel))

        pc_posed = pc_new_gt @ R_rel.T + T_rel
        nm_posed = nm_new_gt @ R_rel.T

        order = np_rng.permutation(len(pc_posed))

        pc_posed = pc_posed[order]
        nm_posed = nm_posed[order]
        pointclouds_gt[start:end] = pointclouds_gt[start:end][order]
        pointclouds_normals_gt[start:end] = pointclouds_normals_gt[start:end][order]
        point_bone_ids[start:end] = point_bone_ids[start:end][order]

        pointclouds.append(pc_posed)
        pointclouds_normals.append(nm_posed)

    pointclouds = np.concatenate(pointclouds).astype(np.float32)
    pointclouds_normals = np.concatenate(pointclouds_normals).astype(np.float32)

    pointclouds, translation_final = recenter_pc(pointclouds)
    pointclouds, scales = rescale_pc(pointclouds)
    pointclouds_gt = (pointclouds_gt - translation_final) / scales

    quaternions, translations_rec = [], []

    for R_rel, T_rel in forward_transforms:
        R_rec = R_rel.T
        T_rec = ((translation_final - T_rel) @ R_rel - translation_final) / scales

        q_xyzw = SciPyRot.from_matrix(R_rec).as_quat()
        q_rec = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        quaternions.append(q_rec)
        translations_rec.append(T_rec.flatten().astype(np.float32))

    quaternions = np.stack(quaternions).astype(np.float32)
    translations = np.stack(translations_rec).astype(np.float32)

    return pointclouds_gt, pointclouds_normals_gt, point_bone_ids, pointclouds, pointclouds_normals, quaternions, translations
