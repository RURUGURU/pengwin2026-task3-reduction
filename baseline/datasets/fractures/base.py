from typing import Literal, List, Tuple, Union
from torch.utils.data import Dataset

import numpy as np
import random
import os
import hashlib

class FragmentDataBase(Dataset):
    """Base Dataset for 3D Bone Fracture Assembly."""

    def __init__(
            self,
            split: Literal["train", "val", "test"] = "train",
            data_root: str = "./data",
            data_category: Union[str, List[str]] = "all",
            min_parts: int = 2,
            max_parts: int = 50,
            max_bone: int = 3,
            random_anchor: bool = False,
            merge_prob: float = 0.15,
            frac_prob: float = 0.70,
            drop_prob: float = 0.1
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split

        if isinstance(data_category, str):
            self.data_category = [data_category.lower()]
        else:
            self.data_category = [c.lower() for c in data_category]

        self.min_parts = min_parts
        self.max_parts = max_parts
        self.max_bone = max_bone

        self.random_anchor = random_anchor
        self.merge_prob = merge_prob
        self.frac_prob = frac_prob
        self.drop_prob = drop_prob

        self.BONE_TYPE_MAP = {"SA": 0, "LI": 1, "RI": 2}

        if not os.path.exists(self.data_root):
            print(f"[Warning] Data root does not exist: {self.data_root}")

        self.fracture_list, self.bone_type_list, self.template_list, self.sample_name_list = self.get_data_list()

        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"Split: {self.split}, "
                  f"Total samples: {len(self.fracture_list)},"
                  f"Bone types: {len(set(self.bone_type_list))},"
                  f"Templates: {len(set(t for t in self.template_list if t is not None))}")

        self.sample_dict = self.build_sample_dict()
        self.index_map = self.build_index_map()
        self.reshuffle()

    def build_sample_dict(self):
        sample_dict = {}
        for frac_path, bone, tmpl, sample in zip(
                self.fracture_list, self.bone_type_list, self.template_list, self.sample_name_list
        ):
            if sample not in sample_dict:
                sample_dict[sample] = {}
            if bone not in sample_dict[sample]:
                sample_dict[sample][bone] = {
                    "fractures": [],
                    "template": tmpl
                }
            sample_dict[sample][bone]["fractures"].append(frac_path)

        for sample in sample_dict:
            for bone in sample_dict[sample]:
                sample_dict[sample][bone]["fractures"] = sorted(
                    sample_dict[sample][bone]["fractures"]
                )
        return sample_dict

    def _extract_bone_type(self, bone_name: str) -> str:
        bone_name = bone_name.upper()
        if "_" in bone_name:
            return bone_name.split("_")[-1]
        return bone_name

    def build_index_map(self):
        """Build task pool for pelvis bone combinations."""
        self.task_pools = {"pelvis": []}
        pelvis_combo = ["SA", "LI", "RI"]

        for sample_name, bones_dict in self.sample_dict.items():
            avail_bones = list(bones_dict.keys())
            avail_bone_types = [self._extract_bone_type(b) for b in avail_bones]
            first_bone = next(iter(bones_dict.values()))
            num_cases = len(first_bone["fractures"])

            for frac_idx in range(num_cases):
                if all(c_type in avail_bone_types for c_type in pelvis_combo):
                    target_bone_names = []
                    for c_type in pelvis_combo:
                        idx = avail_bone_types.index(c_type)
                        target_bone_names.append(avail_bones[idx])
                    self.task_pools["pelvis"].append((sample_name, frac_idx, target_bone_names))

        return []

    def get_data_list(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        fracture_list, bone_type_list, template_list, sample_name_list = [], [], [], []
        train_ratio = 0.90

        if not os.path.exists(self.data_root):
            return [], [], [], []

        search_root = self.data_root
        all_samples = sorted([d for d in os.listdir(search_root) if os.path.isdir(os.path.join(search_root, d))])
        num_samples = len(all_samples)
        if num_samples == 0:
            return [], [], [], []

        split_idx = int(num_samples * train_ratio)
        if self.split == 'train':
            target_samples = all_samples[:split_idx]
        elif self.split == 'val':
            target_samples = all_samples[split_idx:]
        else:
            target_samples = all_samples

        for sample_name in target_samples:
            sample_path = os.path.join(search_root, sample_name)
            source_prefix = os.path.basename(self.data_root)
            unique_sample_id = f"{source_prefix}_{sample_name}"

            for bone_name in sorted(os.listdir(sample_path)):
                bone_path = os.path.join(sample_path, bone_name)
                if not os.path.isdir(bone_path): continue

                template_path = os.path.join(bone_path, "template.npz")
                if not os.path.exists(template_path): template_path = None

                fractures_root = os.path.join(bone_path, "fractures")
                if not os.path.exists(fractures_root): continue

                frac_files = sorted([f for f in os.listdir(fractures_root) if f.endswith(".npz")])
                for frac_file in frac_files:
                    frac_path = os.path.join(fractures_root, frac_file)
                    fracture_list.append(frac_path)
                    bone_type_list.append(bone_name)
                    template_list.append(template_path)
                    sample_name_list.append(unique_sample_id)

        return fracture_list, bone_type_list, template_list, sample_name_list

    def reshuffle(self):
        """Reshuffle fracture samples for data augmentation."""
        for sample in self.sample_dict.values():
            for bone_name, data in sample.items():
                random.shuffle(data["fractures"])

        pelvis_tasks = self.task_pools.get("pelvis", [])
        current_epoch_tasks = list(pelvis_tasks)

        if self.split not in ['val', 'test']:
            random.shuffle(current_epoch_tasks)
        else:
            rng = random.Random(42)
            rng.shuffle(current_epoch_tasks)

        self.index_map = current_epoch_tasks

    def bone_type_to_id(self, bone_type: str) -> int:
        key = bone_type.upper()
        return self.BONE_TYPE_MAP.get(key, 99)

    def shuffle_fragments(self, data_dict, np_rng):
        num_fragments = data_dict["num_parts"]
        if num_fragments <= 1: return data_dict

        points = data_dict["pointclouds_gt"]
        normals = data_dict["pointclouds_normals_gt"]
        offsets = data_dict["offsets"]

        fragments_points, fragments_normals = [], []
        for i in range(num_fragments):
            start_idx, end_idx = offsets[i], offsets[i + 1]
            fragments_points.append(points[start_idx:end_idx])
            fragments_normals.append(normals[start_idx:end_idx])

        new_order = np_rng.permutation(num_fragments)

        data_dict["pointclouds_gt"] = np.vstack([fragments_points[i] for i in new_order])
        data_dict["pointclouds_normals_gt"] = np.vstack([fragments_normals[i] for i in new_order])
        data_dict["offsets"] = np.cumsum([0] + [fragments_points[i].shape[0] for i in new_order])

        return data_dict

    def _get_rng(self, base_path: str):
        if self.split in ['val', 'test']:
            seed = int(hashlib.md5(base_path.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
            rng = random.Random(seed)
            np_rng = np.random.RandomState(seed)
            return rng, np_rng
        else:
            return random, np.random

    def get_frac_data(self, frac_path):
        """Load fracture data from npz file."""
        rng, np_rng = self._get_rng(frac_path)

        with np.load(frac_path) as data:
            pointclouds_gt = data["points"]
            pointclouds_normals_gt = data["normals"]
            offsets = data["offsets"]
        num_parts = len(offsets) - 1

        if np_rng.rand() < self.drop_prob:
            if num_parts >= 2:
                drop_num = 1 if num_parts == 2 else np_rng.choice([1, 2])
                drop_parts_aug = np_rng.choice(num_parts, size=drop_num, replace=False)
                mask = np.ones(pointclouds_gt.shape[0], dtype=bool)
                for p in drop_parts_aug:
                    mask[offsets[p]:offsets[p + 1]] = False
                pointclouds_gt = pointclouds_gt[mask]
                pointclouds_normals_gt = pointclouds_normals_gt[mask]
                new_offsets_aug = [0]
                for i in range(num_parts):
                    if i not in drop_parts_aug:
                        start, end = offsets[i], offsets[i + 1]
                        new_offsets_aug.append(new_offsets_aug[-1] + (end - start))
                offsets = np.array(new_offsets_aug, dtype=offsets.dtype)
                num_parts = len(offsets) - 1

        name_part = split_path_all_levels(frac_path)
        boneType = name_part[-3]
        samplename = name_part[-4] + "_" + name_part[-1] + "_" + name_part[-3]

        frac_data = {
            "path": frac_path, "name": samplename, "num_parts": num_parts,
            "pointclouds_gt": pointclouds_gt, "pointclouds_normals_gt": pointclouds_normals_gt,
            "offsets": offsets, "boneType": boneType,
        }

        return self.shuffle_fragments(frac_data, np_rng)

    def get_template_data(self, template_path, evenness=None):
        """Load template data from npz file."""
        if not template_path or not os.path.exists(template_path):
            raise FileNotFoundError(f"Template path {template_path} does not exist.")

        with np.load(template_path) as data:
            pointclouds_gt = data["points"]
            pointclouds_normals_gt = data["normals"]
            offsets = data["offsets"]

        name_part = split_path_all_levels(template_path)
        boneType = name_part[-2]
        samplename = name_part[-3] + "_" + name_part[-1] + "_" + name_part[-2]

        if evenness == 1:
            return pointclouds_gt
        else:
            return {
                "path": template_path, "name": samplename, "num_parts": 1,
                "pointclouds_gt": pointclouds_gt, "pointclouds_normals_gt": pointclouds_normals_gt,
                "offsets": offsets, "boneType": boneType,
            }

    def transform(self, data: dict):
        raise NotImplementedError

    def __getitem__(self, index):
        sample_name, base_frac_idx, target_bones = self.index_map[index]
        bone_dict = self.sample_dict[sample_name]

        all_points, all_normals, all_bone_types = [], [], []
        all_points_per_pat, all_points_per_bone = [], []
        all_num_parts = 0
        all_point_bone_ids = []
        all_anon_bone_ids_per_part = []

        rng, np_rng = self._get_rng(f"{sample_name}_{index}")

        for i, bone_name in enumerate(target_bones):
            data = bone_dict[bone_name]

            if np_rng.rand() < self.frac_prob:
                safe_frac_idx = base_frac_idx % len(data["fractures"])
                frac_path = data["fractures"][safe_frac_idx]
                part = self.get_frac_data(frac_path)
            else:
                template_path = data["template"]
                part = self.get_template_data(template_path)

            pts = part["pointclouds_gt"]
            normals = part["pointclouds_normals_gt"]
            num_parts = part["num_parts"]
            offsets = part["offsets"]

            boneType_clean = self._extract_bone_type(part["boneType"])
            boneID = self.bone_type_to_id(boneType_clean)
            fracType_list = [boneID] * num_parts

            all_points.append(pts)
            all_normals.append(normals)
            all_points_per_pat.extend(offsets[1:] - offsets[:-1])
            all_points_per_bone.append(len(pts))
            all_bone_types.extend(fracType_list)

            all_point_bone_ids.append(np.full(len(pts), i, dtype=np.int32))
            all_anon_bone_ids_per_part.extend([i] * num_parts)

            all_num_parts += num_parts

        frac_dict = {
            "name": f"{sample_name}_{base_frac_idx}_" + "_".join([self._extract_bone_type(b) for b in target_bones]),
            "pointclouds_gt": np.concatenate(all_points, axis=0),
            "pointclouds_normals_gt": np.concatenate(all_normals, axis=0),
            "boneType": all_bone_types,
            "points_per_part": all_points_per_pat,
            "points_per_bone": all_points_per_bone,
            "num_parts": all_num_parts,
            "point_bone_ids": np.concatenate(all_point_bone_ids, axis=0),
            "anon_bone_ids_per_part": all_anon_bone_ids_per_part,
        }

        frac_dict = remove_extra_parts(frac_dict, max_parts=self.max_parts)
        frac_dict = self.transform(frac_dict, rng=rng, np_rng=np_rng)

        return frac_dict

    def __len__(self):
        return len(self.index_map)

# ================= Helper Functions =================
def remove_extra_parts(frac_dict, max_parts=40):
    num_parts = frac_dict["num_parts"]
    if num_parts <= max_parts:
        return frac_dict

    points_per_part = np.array(frac_dict["points_per_part"])

    drop_count = num_parts - max_parts
    drop_part_indices = np.argsort(points_per_part)[:drop_count]

    keep_part_mask = np.ones(num_parts, dtype=bool)
    keep_part_mask[drop_part_indices] = False

    offsets = np.cumsum(np.insert(points_per_part, 0, 0))
    keep_point_mask = np.ones(len(frac_dict["pointclouds_gt"]), dtype=bool)
    for idx in drop_part_indices:
        keep_point_mask[offsets[idx]:offsets[idx + 1]] = False

    frac_dict["pointclouds_gt"] = frac_dict["pointclouds_gt"][keep_point_mask]
    frac_dict["pointclouds_normals_gt"] = frac_dict["pointclouds_normals_gt"][keep_point_mask]
    frac_dict["point_bone_ids"] = frac_dict["point_bone_ids"][keep_point_mask]

    frac_dict["points_per_part"] = points_per_part[keep_part_mask].tolist()
    frac_dict["boneType"] = [frac_dict["boneType"][i] for i, keep in enumerate(keep_part_mask) if keep]

    new_anon_bone_ids = [frac_dict["anon_bone_ids_per_part"][i] for i, keep in enumerate(keep_part_mask) if keep]
    frac_dict["anon_bone_ids_per_part"] = new_anon_bone_ids

    bone_points = {}
    for anon_id, pts_count in zip(new_anon_bone_ids, frac_dict["points_per_part"]):
        bone_points[anon_id] = bone_points.get(anon_id, 0) + pts_count

    frac_dict["points_per_bone"] = list(bone_points.values())
    frac_dict["num_parts"] = max_parts

    return frac_dict


def split_path_all_levels(path):
    parts = []
    while True:
        path, tail = os.path.split(path)
        if tail != "":
            parts.append(tail)
        else:
            if path != "": parts.append(path)
            break
    return parts[::-1]