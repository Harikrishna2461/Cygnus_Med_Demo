import supervision as sv
import os
import yaml
import json

# === Paths ===
DATA_DIR = "data"
BASE_NAME = "data_coco"
OUTPUT_NAME = "data_yolo"

BASE_DIR = f"{DATA_DIR}/{BASE_NAME}"
OUTPUT_DIR = f"{DATA_DIR}/{OUTPUT_NAME}"

# Input splits (what exists in your data_coco)
INPUT_SPLITS = ["train", "valid", "test"]

# Map input folder → output folder name
SPLIT_MAP = {
    "train": "train",
    "valid": "val",  # rename on output
    "test": "test"
}

# === Prepare output structure ===
for out_split in SPLIT_MAP.values():
    os.makedirs(f"{OUTPUT_DIR}/images/{out_split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{out_split}", exist_ok=True)

# === Convert each split ===
for in_split, out_split in SPLIT_MAP.items():
    ann_path = os.path.join(BASE_DIR, in_split, "_annotations.coco.json")
    img_dir = os.path.join(BASE_DIR, in_split)

    if not os.path.exists(ann_path):
        print(f"⚠️ No annotations found for {in_split} at {ann_path}, skipping...")
        continue

    print(f"🔄 Converting {in_split} → {out_split} ...")

    dataset = sv.DetectionDataset.from_coco(
        annotations_path=ann_path,
        images_directory_path=img_dir,
    )

    dataset.as_yolo(
        annotations_directory_path=f"{OUTPUT_DIR}/labels/{out_split}/",
        images_directory_path=f"{OUTPUT_DIR}/images/{out_split}/"
    )

print("✅ Conversion complete for all splits!")

# === Create data.yaml ===
# Load class names from any available annotation file
for in_split in INPUT_SPLITS:
    ann_path = os.path.join(BASE_DIR, in_split, "_annotations.coco.json")
    if os.path.exists(ann_path):
        with open(ann_path, "r") as f:
            coco_data = json.load(f)
        sorted_cats = sorted(coco_data["categories"], key=lambda x: x["id"])
        id_to_name = {i: cat["name"] for i, cat in enumerate(sorted_cats)}
        break
else:
    raise FileNotFoundError("No COCO annotation files found to extract class names.")

data_yaml = {
    "path": OUTPUT_NAME,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": len(id_to_name),
    "names": id_to_name,
}

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, sort_keys=False)

print(f"📄 data.yaml created successfully at {yaml_path}")
