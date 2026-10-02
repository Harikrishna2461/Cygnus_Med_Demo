Ultrasound Vein Segmentation – Handover
=======================================

Overview
--------
- Interactive PySide6 application for annotating ultrasound videos, propagating SAM 2 segmentations, and exporting training data.
- Core flow: load video → annotate sparse points per vein → run SAM 2 propagation → review overlays → export frames, masks, metadata.
- Scripts convert saved segmentations into COCO and YOLO datasets for downstream detection projects.
- Requires Python 3.12+.

Repository Layout
-----------------

```
project_segmentation/
├─ main.py                  # App entry point (creates QApplication + UltrasoundApp)
├─ gui/                     # GUI widgets and interactions
├─ segmentation/            # SAM 2 video wrapper and domain enums (veins and SAM models)
├─ converters/              # Dataset post-processing utilities (create COCO and YOLO datasets)
├─ sam2/                    # SAM2 library
├─ data/                    # Working data (Input: videos; Output: frames, segmentations, metadata)
├─ tests/                   # Pytest-based smoke tests and fixtures
├─ utils.py                 # Helpers for the app
└─ pyproject.toml           # Dependencies and optional Torch extras
```

| Directory / File | Purpose |
| --- | --- |
| `main.py` | Bootstraps the GUI by instantiating `gui.main_window.UltrasoundApp`. |
| `gui/` | Custom PySide6 widgets: layout, playback, annotation, crop tooling. |
| `segmentation/` | `SamVideoSegmenter` wrapper over SAM 2 video predictor, plus model and domain enums. |
| `sam2/` | Import and config for SAM 2. |
| `converters/` | Offline scripts to turn saved `.npz` masks into COCO/YOLO datasets and augment data. |
| `data/` | Runtime artifacts: raw videos, extracted frames, per-frame masks, metadata, converted datasets. |
| `tests/` | Pytest-based minimal coverage and fixtures. |
| `utils.py` | Helpers (OpenCV ↔ Qt conversions). |
| `pyproject.toml` | Dependency manifest. |

Application Architecture
------------------------
- GUI shell (`gui.main_window.UltrasoundApp`)
  - Manages video playback (`cv2.VideoCapture`) with manual frame navigation and FPS control.
  - Collects user annotations (positive/negative points per `Vein` enum) through `AnnotatedImageLabel` signals.
  - Supports crop selection to focus segmentation on a region of interest; crop region is saved in metadata during export.
  - Delegates segmentation to `SamVideoSegmenter`, keeping frames and overlays in memory for side-by-side review.

- Segmentation engine (`segmentation.sam_segmentation.SamVideoSegmenter`)
  - Downloads SAM 2 checkpoints if missing using `SamModel` enum metadata (size suffix, config name, URL).
  - Extracts video frames into `data/frames/<video_name>/` on first run and initializes the SAM video predictor.
  - Accepts sparse prompts (points) per object id and runs propagation to obtain per-frame binary masks.
  - Provides utilities for rendering masked frames and exporting compressed `.npz` stacks plus JSON metadata.

- Domain enums
  - `segmentation.veins.Vein`: clinical vein classes with integer ids and RGB colors used for overlay.
  - `segmentation.sam_models.SamModel`: SAM 2 checkpoint definitions (tiny/small/base+/large).

- Shared widgets
  - `gui.annotated_image_label.AnnotatedImageLabel` wraps `QLabel` to capture clicks, render annotation ids, and overlay crop rectangles.

- Utilities
  - `utils.cv_to_qt_pixmap` handles OpenCV → Qt conversions (RGB) for GUI display.

Operational Data Flow
---------------------
1. Video load
   - Reads video frames into memory (`self.frames`), resets UI state, initializes crop region, and primes slider and counters.
2. Model selection
   - User picks a SAM 2 variant via combo bound to `SamModel` enum; `load_sam_model` extracts frames (if required) and initializes predictor state.
3. Annotation
   - App stores per-vein point/label/frame triplets in `self.annotations` and tracks annotated frames.
   - Crop mode collects bounding points for `self.crop_region` without generating prompts.
4. Segmentation
   - `process_video` replays stored prompts and runs full propagation to populate `self.segmented_frames` for playback.
   - Overlays are shown via `cv_to_qt_pixmap`.
5. Export
   - `save_segmentation` writes `data/segmentations/<video>.npz` and `data/metadata/<video>.json` (crop extent, frame count).
   - Masks outside the crop are zeroed. Load the `.npz` to access per-frame mask stacks (shape: n_classes × H × W).

Dataset Conversion Tooling (`converters/`)
-----------------------------------------
- `to_coco.py`: Converts saved frame folders and `.npz` mask archives into COCO detection/segmentation splits (train/valid/test). Optionally cleans masks via morphological ops.
- `coco_to_yolo.py`: Re-exports COCO splits to YOLO format, generating standard `data.yaml`.
- `coco_to_coco_mono.py`: Collapses multi-class annotations into a single “vein” category for monoclass training.
- `converters/utils.py`: Shared helpers for mask cleaning and polygon extraction.

Data Directories
----------------
- `data/videos/`: Source ultrasound clips loaded in the GUI; not stored in the repository.
- `data/frames/<video>/`: Extracted per-frame JPGs (auto-generated).
- `data/segmentations/`: Compressed `.npz` files keyed by frame index containing stacked masks (channel order = vein id + 1).
- `data/metadata/`: JSON per video containing crop region and frame count.
- `data/data_coco*/`, `data/data_yolo*/`: Outputs from converters for downstream training.

SAM 2 Integration (`sam2/`)
---------------------------
- Config directories (`sam2/configs/sam2.1/`) host Hydra YAMLs referenced by `SamModel`.
- Expect large checkpoint downloads (~several hundred MB) on first use; checkpoints are stored under `segmentation/checkpoints/`.

Dependencies & Environment
--------------------------
- Listed in `pyproject.toml`:
  - Core: `PySide6`, `opencv-python`, `matplotlib`, `hydra-core`, `iopath`, `tqdm`, `supervision`, `scikit-learn`.
  - Optional PyTorch extras for different CUDA versions.
  - Dev dependencies include `pytest`.
- Typical setup commands:
  - With GPU extras:
    ```
    uv sync --extra cu129
    uv run main.py
    ```
  - CPU-only:
    ```
    uv sync
    uv run main.py
    ```
  - Without dev dependencies:
    ```
    uv sync --extra cu129 --no-dev
    uv run main.py
    ```

Operational Tips
----------------
- GPU selection: `SamVideoSegmenter` uses CUDA if available (`torch.cuda.is_available()`).
- Memory usage: `load_video` currently reads full video into RAM; very long clips may require a streaming refactor.
- Crop defaults: Initial crop approximates likely vein regions; adjust per video via crop mode button, which updates metadata export.
- Positive vs negative prompts: Use positive clicks to mark vein interiors and negative clicks for background to help SAM disambiguate structures.