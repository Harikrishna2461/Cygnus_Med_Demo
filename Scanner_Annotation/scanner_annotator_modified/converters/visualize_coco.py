#%%
import os
import json
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
# Add the project root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from segmentation.veins import Vein

def plot_annotations(image_path, annotation_path, use_supercategory=False):
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    # Convert color for Matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Load COCO annotations
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found at {annotation_path}")
        return
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Extract the filename from the image path
    filename = os.path.basename(image_path)

    # Find the image_id corresponding to the filename
    image_id = None
    for image in coco_data['images']:
        if image['file_name'] == filename:
            image_id = image['id']
            break
    if image_id is None:
        print(f"No image entry found for filename: {filename}")
        return

    # Find annotations for the specified image_id
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    if not annotations:
        print(f"No annotations found for image: {filename}")
        return

    # Define colors for each category (using Vein enum colors)
    category_colors = {
        vein.id + 1: vein.color.tolist() for vein in Vein
    }
    # Default color for supercategory (e.g., white)
    supercategory_color = (255, 255, 255)

    # Plot each annotation
    for ann in annotations:
        if use_supercategory:
            # Use supercategory (id 0) and its color
            category_id = 0
            color = supercategory_color
        else:
            # Use the original category_id and its color
            category_id = ann['category_id']
            color = category_colors[category_id]
        bbox = ann['bbox']
        segmentation = ann['segmentation']
        # Draw bounding box (thickness=1 for thinner lines)
        x, y, w, h = map(int, bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        # Draw segmentation polygon (thickness=1 for thinner lines)
        for seg in segmentation:
            points = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.polylines(img, [points], isClosed=True, color=color, thickness=1)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Image: {filename} | {'Supercategory' if use_supercategory else 'Subcategories'}")
    plt.show()

#%%
# Example usage
if __name__ == "__main__":
    filename = "1764147453_0002_00000.jpg"  # Replace with the filename you want to visualize
    data_dir = "data/data_coco"  # Replace with your output directory
    split = "train"  # Replace with "valid" or "test" if needed

    # Paths
    image_path = os.path.join(data_dir, split, filename)
    annotation_path = os.path.join(data_dir, split, "_annotations.coco.json")

    # Debug prints
    print(f"Looking for image at: {image_path}")
    print(f"Looking for annotations at: {annotation_path}")

    # Plot with subcategories (default)
    plot_annotations(image_path, annotation_path, use_supercategory=False)

    # Plot with supercategory
    plot_annotations(image_path, annotation_path, use_supercategory=True)

# %%
