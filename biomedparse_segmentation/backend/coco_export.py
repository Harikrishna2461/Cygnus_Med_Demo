"""
COCO 1.0 export of per-frame vein masks, packaged as a zip CVAT can import
directly as an "Images" task in the "COCO 1.0" format:

    <zip>/
      images/
        frame_000000.jpg
        frame_000001.jpg
        ...
      annotations/
        instances_default.json

Masks are converted to polygon segmentations (cv2.findContours), not RLE —
polygons are what CVAT's COCO importer turns back into editable mask/polygon
shapes, and they need no pycocotools dependency to write.
"""
import io
import json
import os
import zipfile

import cv2
import numpy as np
from PIL import Image

CATEGORY_ID_VEIN = 1

COCO_CATEGORIES = [
    {"id": CATEGORY_ID_VEIN, "name": "vein", "supercategory": "anatomy"},
]


def mask_to_polygons(mask: np.ndarray, min_points: int = 3):
    """uint8 {0,1} mask -> list of COCO polygons (each a flat [x0,y0,x1,y1,...] float list)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) < min_points:
            continue
        poly = cnt.reshape(-1, 2).astype(np.float64).flatten().tolist()
        if len(poly) >= 6:  # need >=3 (x,y) pairs
            polygons.append(poly)
    return polygons


def polygon_bbox_area(polygons):
    """Compute COCO bbox [x,y,w,h] and area from a list of flat polygons."""
    all_pts = np.concatenate([np.array(p).reshape(-1, 2) for p in polygons], axis=0)
    x0, y0 = all_pts.min(axis=0)
    x1, y1 = all_pts.max(axis=0)
    bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    area = 0.0
    for p in polygons:
        pts = np.array(p).reshape(-1, 2).astype(np.float32)
        area += abs(cv2.contourArea(pts))
    return bbox, float(area)


class CocoVideoAnnotationBuilder:
    """
    Accumulates one COCO "image" + zero-or-more "annotation" entries per
    video frame, plus the frame's JPEG bytes, then writes everything out as
    a CVAT-COCO-1.0-compatible zip.
    """

    def __init__(self, video_name: str, fps: float, width: int, height: int):
        self.video_name = video_name
        self.fps = fps
        self.width = width
        self.height = height
        self.images = []
        self.annotations = []
        self._frame_jpegs = {}  # file_name -> jpeg bytes
        self._next_ann_id = 1

    def add_frame(self, frame_index: int, frame_rgb: np.ndarray, vein_mask: np.ndarray):
        file_name = f"frame_{frame_index:06d}.jpg"
        image_id = frame_index + 1  # COCO ids are 1-based by convention

        self.images.append({
            "id": image_id,
            "file_name": file_name,
            "width": self.width,
            "height": self.height,
            # extra (non-standard, ignored by strict parsers) fields kept for traceability
            "frame_index": frame_index,
            "timestamp_sec": round(frame_index / self.fps, 6) if self.fps > 0 else 0.0,
        })

        if vein_mask is not None and vein_mask.max() > 0:
            # one annotation per connected vein blob
            n, labels = cv2.connectedComponents(vein_mask.astype(np.uint8), connectivity=8)
            for blob_id in range(1, n):
                blob_mask = (labels == blob_id).astype(np.uint8)
                polygons = mask_to_polygons(blob_mask)
                if not polygons:
                    continue
                bbox, area = polygon_bbox_area(polygons)
                self.annotations.append({
                    "id": self._next_ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID_VEIN,
                    "segmentation": polygons,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                self._next_ann_id += 1

        buf = io.BytesIO()
        Image.fromarray(frame_rgb).save(buf, format='JPEG', quality=95)
        self._frame_jpegs[file_name] = buf.getvalue()

    def instances_json(self) -> dict:
        return {
            "info": {
                "description": f"BiomedParse vein segmentation — {self.video_name}",
                "video_name": self.video_name,
                "fps": self.fps,
                "frame_count": len(self.images),
            },
            "licenses": [],
            "images": self.images,
            "annotations": self.annotations,
            "categories": COCO_CATEGORIES,
        }

    def write_zip(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('annotations/instances_default.json', json.dumps(self.instances_json(), indent=2))
            for file_name, jpeg_bytes in self._frame_jpegs.items():
                zf.writestr(f'images/{file_name}', jpeg_bytes)
        return out_path
