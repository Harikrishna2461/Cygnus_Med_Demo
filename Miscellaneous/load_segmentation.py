import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np


def install_numpy_pickle_compat_aliases():
    """Allow NumPy 1.x to unpickle arrays saved by NumPy 2.x."""
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_core_multiarray
        import numpy.core.numeric as numpy_core_numeric
    except Exception:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_core_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)


def load_npz_safe(filepath):
    install_numpy_pickle_compat_aliases()
    return np.load(filepath, allow_pickle=True)


def unwrap_object_scalar(value):
    arr = np.asarray(value)
    while isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        arr = arr.item()
    return arr


def normalize_mask(mask):
    arr = unwrap_object_scalar(mask)
    arr = np.asarray(arr)
    if arr.dtype == object:
        arr = arr.astype(np.uint8)
    return (arr > 0).astype(np.uint8)


def parse_frame_idx(key):
    try:
        return int(key)
    except (TypeError, ValueError):
        key_str = str(key)
        if key_str.startswith("frame_"):
            try:
                return int(key_str.split("_", 1)[1])
            except ValueError:
                return None
        return None


def parse_obj_id_from_filename(filename):
    match = re.match(r"^obj_(\d+)_chunk_\d+\.npz$", filename)
    if match:
        return int(match.group(1))
    return None


def get_file_format(filename):
    if re.match(r"^obj_\d+_chunk_\d+\.npz$", filename):
        return "object_based"
    if re.match(r"^chunk_\d+\.npz$", filename):
        return "frame_based"
    return "unknown"


def extract_obj_ids_from_entry(entry, fallback_obj_id=None):
    value = unwrap_object_scalar(entry)
    if isinstance(value, dict):
        obj_ids = []
        for key in value.keys():
            try:
                obj_ids.append(int(key))
            except (TypeError, ValueError):
                continue
        return obj_ids
    if fallback_obj_id is not None:
        return [fallback_obj_id]
    return []


def get_mask_for_obj(entry, obj_id):
    value = unwrap_object_scalar(entry)
    if isinstance(value, dict):
        if obj_id in value:
            return normalize_mask(value[obj_id])
        key_str = str(obj_id)
        if key_str in value:
            return normalize_mask(value[key_str])
        return None
    return normalize_mask(value)


class SegmentationMaskLoader:
    """Load masks from a folder containing `chunk_*.npz` and/or `obj_*_chunk_*.npz` files."""

    def __init__(self, mask_dir=None):
        self.mask_dir = None
        self.frame_index = {}
        self._npz_cache = {}
        if mask_dir is not None:
            self.load(mask_dir)

    def close(self):
        for npz_data in self._npz_cache.values():
            try:
                npz_data.close()
            except Exception:
                pass
        self._npz_cache = {}

    def clear(self):
        self.close()
        self.mask_dir = None
        self.frame_index = {}

    def load(self, mask_dir):
        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask folder not found: {mask_dir}")

        frame_index = self.build_frame_mask_index(mask_dir)
        self.clear()
        self.mask_dir = mask_dir
        self.frame_index = frame_index
        return len(frame_index)

    def build_frame_mask_index(self, mask_dir):
        frame_index = defaultdict(dict)

        for filename in sorted(os.listdir(mask_dir)):
            if not filename.endswith(".npz"):
                continue

            filepath = os.path.join(mask_dir, filename)
            if not os.path.isfile(filepath):
                continue

            fallback_obj_id = parse_obj_id_from_filename(filename)
            file_format = get_file_format(filename)
            if file_format == "unknown":
                continue

            with load_npz_safe(filepath) as data:
                for key in data.files:
                    frame_idx = parse_frame_idx(key)
                    if frame_idx is None:
                        continue

                    entry = data[key]
                    if file_format == "frame_based":
                        obj_ids = extract_obj_ids_from_entry(entry)
                    else:
                        obj_ids = extract_obj_ids_from_entry(
                            entry,
                            fallback_obj_id=fallback_obj_id,
                        )

                    for obj_id in obj_ids:
                        frame_index[frame_idx][obj_id] = (filepath, key)

        return dict(frame_index)

    def get_cached_npz(self, filepath):
        cached = self._npz_cache.get(filepath)
        if cached is None:
            cached = load_npz_safe(filepath)
            self._npz_cache[filepath] = cached
        return cached

    def list_frame_indices(self):
        return sorted(self.frame_index.keys())

    def list_object_ids(self, frame_idx):
        return sorted(self.frame_index.get(frame_idx, {}).keys())

    def has_frame(self, frame_idx):
        return frame_idx in self.frame_index

    def get_frame_masks(self, frame_idx):
        """Return one frame's masks as `{obj_id: mask}`."""
        obj_entries = self.frame_index.get(frame_idx, {})
        frame_masks = {}
        for obj_id, (filepath, key) in obj_entries.items():
            npz_data = self.get_cached_npz(filepath)
            mask = get_mask_for_obj(npz_data[key], obj_id)
            if mask is not None:
                frame_masks[obj_id] = mask
        return frame_masks

    def get_object_mask(self, frame_idx, obj_id):
        """Return a single object's mask for one frame."""
        obj_entries = self.frame_index.get(frame_idx, {})
        entry = obj_entries.get(obj_id)
        if entry is None:
            return None

        filepath, key = entry
        npz_data = self.get_cached_npz(filepath)
        return get_mask_for_obj(npz_data[key], obj_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load masks from a folder containing chunk/object npz files.",
    )
    parser.add_argument(
        "--mask_dir",
        required=True,
        help="Folder containing `chunk_*.npz` and/or `obj_*_chunk_*.npz` files.",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=None,
        help="Optional frame index to inspect.",
    )
    parser.add_argument(
        "--obj_id",
        type=int,
        default=None,
        help="Optional object id to inspect together with --frame_idx.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    loader = SegmentationMaskLoader(args.mask_dir)

    try:
        frame_indices = loader.list_frame_indices()
        print(f"Mask dir    : {args.mask_dir}")
        print(f"Frame count : {len(frame_indices)}")

        if frame_indices:
            print(f"First frame : {frame_indices[0]}")
            print(f"Last frame  : {frame_indices[-1]}")

        if args.frame_idx is None:
            return

        frame_masks = loader.get_frame_masks(args.frame_idx)
        print(f"Frame {args.frame_idx} object ids: {sorted(frame_masks.keys())}")

        if args.obj_id is None:
            for obj_id, mask in sorted(frame_masks.items()):
                print(f"obj {obj_id}: shape={mask.shape}, sum={int(mask.sum())}")
            return

        mask = loader.get_object_mask(args.frame_idx, args.obj_id)
        if mask is None:
            print(f"No mask found for frame {args.frame_idx}, obj {args.obj_id}")
            return

        print(
            f"frame={args.frame_idx}, obj={args.obj_id}, "
            f"shape={mask.shape}, dtype={mask.dtype}, sum={int(mask.sum())}"
        )
    finally:
        loader.close()


if __name__ == "__main__":
    main()
