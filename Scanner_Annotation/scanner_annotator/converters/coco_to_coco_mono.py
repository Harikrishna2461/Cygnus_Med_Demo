import os
import json
import shutil

def replace_categories_with_vein_and_copy_data(data_coco_dir, data_coco_mono_dir):
    """
    Create a new dataset with all classes replaced by "vein" (id: 0).
    Copies frames and modifies annotations.
    """
    # Create the new directory structure
    os.makedirs(os.path.join(data_coco_mono_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_coco_mono_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(data_coco_mono_dir, "test"), exist_ok=True)

    # Define splits
    splits = ["train", "valid", "test"]

    for split in splits:
        # Paths
        src_frames_dir = os.path.join(data_coco_dir, split)
        dst_frames_dir = os.path.join(data_coco_mono_dir, split)
        src_annot_path = os.path.join(data_coco_dir, split, "_annotations.coco.json")
        dst_annot_path = os.path.join(data_coco_mono_dir, split, "_annotations.coco.json")

        # Copy frames
        for frame in os.listdir(src_frames_dir):
            src_frame_path = os.path.join(src_frames_dir, frame)
            dst_frame_path = os.path.join(dst_frames_dir, frame)
            shutil.copy(src_frame_path, dst_frame_path)

        # Load and modify annotations
        with open(src_annot_path, 'r') as f:
            coco_data = json.load(f)

        # Replace categories
        coco_data['categories'] = [
            {
                "supercategory": None,
                "id": 0,
                "name": "vein"
            }
        ]

        # Replace all category_ids in annotations
        for annotation in coco_data['annotations']:
            annotation['category_id'] = 0

        # Save modified annotations
        with open(dst_annot_path, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print(f"✅ Processed {split}: frames copied, annotations modified.")

# Example usage
if __name__ == "__main__":
    data_coco_dir = "data/data_coco"
    data_coco_mono_dir = "data/data_coco_mono"

    replace_categories_with_vein_and_copy_data(data_coco_dir, data_coco_mono_dir)
    print(f"✅ Dataset with monoclass 'vein' created at {data_coco_mono_dir}")
