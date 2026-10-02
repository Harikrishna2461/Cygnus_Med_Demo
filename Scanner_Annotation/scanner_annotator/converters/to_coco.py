#%%
import os
import sys
import json
import cv2
import numpy as np
import random
import shutil
from sklearn.model_selection import train_test_split

# Add the project root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from segmentation.veins import Vein  # Import the Vein enum
from converters.utils import clean_mask, mask_to_polygons

def split_and_convert_to_coco(data_dir, output_dir, val_size=0.15, test_size=0.15, random_seed=42,
                              clean_masks=True, kernel_size=3, iterations=1):
    # Initialize containers for all data
    coco_categories = [
        {
            "supercategory": None,
            "id": 0,
            "name": "vein"
        }
    ]
    coco_categories.extend([
        {
            "supercategory": "vein",
            "id": vein.id + 1,
            "name": vein.label
        }
        for vein in Vein
    ])
    frames_dir = os.path.join(data_dir, "frames")
    seg_dir = os.path.join(data_dir, "segmentations")
    meta_dir = os.path.join(data_dir, "metadata")
    img_id = 1
    ann_id = 1
    all_data = []

    for video_name in sorted(os.listdir(frames_dir)):
        video_path = os.path.join(frames_dir, video_name)
        if not os.path.isdir(video_path):
            continue

        # --- Load metadata (crop region)
        crop_region = None
        meta_path = os.path.join(meta_dir, f"{video_name}.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                crop_region = meta.get("crop_region")

        # --- Load segmentation file
        seg_path = os.path.join(seg_dir, f"{video_name}.npz")
        if not os.path.exists(seg_path):
            print(f"[Warning] No segmentation file for {video_name}, skipping.")
            continue
        seg_data = np.load(seg_path)
        frame_files = sorted(os.listdir(video_path))

        for frame_idx, frame_name in enumerate(frame_files):
            img_path = os.path.join(video_path, frame_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Could not read image: {img_path}")
                continue

            # --- Crop the image if crop_region is available
            if crop_region:
                xmin, ymin, xmax, ymax = crop_region
                img = img[ymin:ymax, xmin:xmax]
                height, width = img.shape[:2]
            else:
                height, width = img.shape[:2]

            # --- Register image entry
            image_info = {
                "video": video_name,
                "frame": frame_name,
                "width": width,
                "height": height,
                "img_id": img_id,
                "crop_region": crop_region,
                "frame_index": frame_idx  # Store frame index for filename
            }

            # --- Load segmentation masks for this frame
            key = str(frame_idx)
            if key not in seg_data:
                img_id += 1
                continue
            frame_masks = seg_data[key]  # shape (n_masks, H, W)
            annotations = []
            for obj_i, mask in enumerate(frame_masks):
                if mask.sum() == 0:
                    continue
                if crop_region:
                    xmin, ymin, xmax, ymax = crop_region
                    mask = mask[ymin:ymax, xmin:xmax]

                # Clean the mask if requested
                if clean_masks:
                    mask = clean_mask(mask, kernel_size=kernel_size, iterations=iterations)

                # Extract polygons from the cleaned mask
                polygons = mask_to_polygons(mask)
                if not polygons:
                    continue

                # For each polygon in the mask, create a separate annotation
                for polygon in polygons:
                    # Create a binary mask for this polygon
                    polygon_mask = np.zeros_like(mask, dtype=np.uint8)
                    polygon_contour = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(polygon_mask, [polygon_contour], 1)

                    # Find bounding box for this polygon
                    ys, xs = np.where(polygon_mask > 0)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    xmin, xmax = xs.min(), xs.max()
                    ymin, ymax = ys.min(), ys.max()
                    area = float(polygon_mask.sum())
                    bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                    category_id = obj_i + 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": category_id,
                        "segmentation": [polygon],  # Single polygon per annotation
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    })
                    ann_id += 1

            if annotations:
                all_data.append({
                    "image": img,
                    "image_info": image_info,
                    "annotations": annotations
                })
                img_id += 1

        seg_data.close()

    # --- Split data into train, val, test
    val_file_set = set()
    with open('val_files.txt', 'r') as val_filenames:
        for line in val_filenames:
            videoname = line.strip()
            val_file_set.add(videoname)
    test_file_set = set()
    with open('test_files.txt', 'r') as test_filenames:
        for line in test_filenames:
            videoname = line.strip()
            test_file_set.add(videoname)
    
    train_data = []
    val_data = []
    test_data = []
    for data in all_data:
        video_info = data["image_info"]["video"]
        if video_info in val_file_set:
            val_data.append(data)
        if video_info in test_file_set:
            test_data.append(data)
        else:
            train_data.append(data)
    random.seed(random_seed)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    # random.shuffle(all_data)
    # train_data, val_test_data = train_test_split(all_data, test_size=(val_size + test_size), random_state=random_seed)
    # val_data, test_data = train_test_split(val, test_size=(test_size/(val_size + test_size)), random_state=random_seed)

    # --- Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "valid"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

    # --- Function to save a split
    def save_split(split_data, split_name):
        images = []
        annotations = []
        a_id = 1
        i_id = 1
        for item in split_data:
            img = item["image"]
            image_info = item["image_info"]
            anns = item["annotations"]
            # Generate filename as videoname_frameindex.jpg
            filename = f"{image_info['video']}_{image_info['frame_index']:05d}.jpg"
            dst_path = os.path.join(output_dir, split_name, filename)
            cv2.imwrite(dst_path, img)
            images.append({
                "id": i_id,
                "file_name": filename,
                "width": image_info["width"],
                "height": image_info["height"]
            })
            for ann in anns:
                annotations.append({
                    "id": a_id,
                    "image_id": i_id,
                    "category_id": ann["category_id"],
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "iscrowd": ann["iscrowd"]
                })
                a_id += 1
            i_id += 1
        coco = {
            "info": {
                "year": 2025,
                "version": "1.0",
                "description": "Vein Segmentation Dataset",
                "contributor": "Contributor",
                "url": "https://yourwebsite.com/dataset",
                "date_created": "2025-10-30"
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)",
                    "url": "https://creativecommons.org/licenses/by-nc/4.0/"
                }
            ],
            "categories": coco_categories,
            "images": images,
            "annotations": annotations
        }
        with open(os.path.join(output_dir, split_name, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f, indent=2)


    # --- Save splits
    save_split(train_data, "train")
    save_split(val_data, "valid")
    save_split(test_data, "test")

    print(f"✅ Dataset split and saved to {output_dir}")

#%%
# Example usage:
if __name__ == "__main__":
    split_and_convert_to_coco(
        data_dir="data",
        output_dir="data/data_coco",
        val_size=0.15,
        test_size=0.05,
        random_seed=42,
        clean_masks=False,  # Enable mask cleaning
        kernel_size=10,     # Kernel size for morphological operations
        iterations=1      # Number of iterations for morphological operations
    )
