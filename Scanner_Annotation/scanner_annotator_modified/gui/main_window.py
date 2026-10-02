import cv2
import torch
import numpy as np
import os
import json
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QFileDialog, QLineEdit, QComboBox, QSlider, QMessageBox
)
from PySide6.QtCore import QTimer, Qt, QFileSystemWatcher
from PySide6.QtGui import QCursor

from utils import cv_to_qt_pixmap
from segmentation.sam_segmentation import SamVideoSegmenter

from segmentation.sam_models import SamModel
from segmentation.veins import Vein

from gui.annotated_image_label import AnnotatedImageLabel


class UltrasoundApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultrasound Vein Segmentation - Prototype")

        # Video and segmentation state
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.playing = False
        self.fps = 60
        self.frames = []                 # list of raw video frames (BGR)
        self.segmented_frames = []       # list of processed / masked frames
        self.current_frame_idx = 0
        self.displayed_frame_idx = 0
        self.segmenter = None
        self.video_path = None
        self.annotations = {}            # {vein_id: {"points":[], "labels":[], "frames":[]}}
        self.annotated_frames = set()    # set of frame indices that have annotations

        # Crop region state
        self.crop_mode = False
        self.crop_first_point = None
        self.crop_region = None  # [xmin, ymin, xmax, ymax]
        # Crop region button
        self.btn_crop_region = QPushButton("Crop region")
        self.btn_crop_region.setCheckable(True)

        # --- UI Elements ---
        # Video display labels
        self.left_image_label = AnnotatedImageLabel(self, interactive=True)
        self.right_image_label = AnnotatedImageLabel(self, interactive=False)
        self.left_image_label.setCursor(QCursor(Qt.CrossCursor))
        self.right_image_label.setCursor(QCursor(Qt.CrossCursor))
        self.left_image_label.setFixedSize(600, 400)
        self.right_image_label.setFixedSize(600, 400)

        # Playback controls
        self.btn_prev_frame = QPushButton("Previous frame")
        self.btn_prev_frame.setEnabled(False)
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_next_frame = QPushButton("Next frame")
        self.btn_next_frame.setEnabled(False)
        self.fps_label = QLabel("FPS:")
        self.fps_label.setFixedWidth(30)
        self.fps_input = QLineEdit(str(self.fps))
        self.fps_input.setFixedWidth(50)
        self.fps_input.editingFinished.connect(self.update_fps)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_counter_label = QLabel("0 / 0")
        self.frame_counter_label.setFixedWidth(80)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        # Video and model controls
        self.btn_load = QPushButton("Load Video")
        self.model_label = QLabel("SAM Model:")
        self.model_label.setFixedWidth(70)
        self.model_combo = QComboBox()
        for model in SamModel:
            self.model_combo.addItem(model.name, model)
        # Default to BASE_PLUS rather than TINY: meaningfully less drift on long/merged
        # clips, and this machine has the headroom for it (see SamVideoSegmenter).
        default_idx = self.model_combo.findText(SamModel.BASE_PLUS.name)
        if default_idx >= 0:
            self.model_combo.setCurrentIndex(default_idx)
        self.sam_model = self.model_combo.currentData()
        self.btn_load_sam = QPushButton("Load SAM")
        self.btn_process = QPushButton("Process Video")
        self.btn_save_seg = QPushButton("Save Segmentation")
        self.btn_load_seg = QPushButton("Load Segmentation")

        # Annotations
        self.btn_positive_annotation = QPushButton("Positive Annotation")
        self.btn_positive_annotation.setCheckable(True)
        self.btn_positive_annotation.setChecked(True)
        self.btn_negative_annotation = QPushButton("Negative Annotation")
        self.btn_negative_annotation.setCheckable(True)
        self.vein_combo = QComboBox()
        for vein in Vein:
            self.vein_combo.addItem(f'{vein.label} : {vein.id}', vein)
        self.current_vein = self.vein_combo.currentData()
        self.btn_clear_annotations = QPushButton("Clear Annotations")
        self.btn_clear_all_annotations = QPushButton("Clear ALL Annotations")
        self.btn_import_annotations = QPushButton("Import Annotations")

        # --- Layouts ---
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.left_image_label)
        video_layout.addWidget(self.right_image_label)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_prev_frame)
        control_layout.addWidget(self.btn_play_pause)
        control_layout.addWidget(self.btn_next_frame)
        control_layout.addWidget(self.fps_label)
        control_layout.addWidget(self.fps_input)

        process_layout = QHBoxLayout()
        process_layout.addWidget(self.btn_load)
        process_layout.addWidget(self.model_label)
        process_layout.addWidget(self.model_combo)
        process_layout.addWidget(self.btn_load_sam)
        process_layout.addWidget(self.btn_process)
        process_layout.addWidget(self.btn_save_seg)
        process_layout.addWidget(self.btn_load_seg)

        annotation_layout = QHBoxLayout()
        annotation_layout.addWidget(self.btn_positive_annotation)
        annotation_layout.addWidget(self.btn_negative_annotation)
        annotation_layout.addWidget(self.vein_combo)
        annotation_layout.addWidget(self.btn_clear_annotations)
        annotation_layout.addWidget(self.btn_clear_all_annotations)
        annotation_layout.addWidget(self.btn_import_annotations)
        # annotation_layout.addWidget(self.btn_crop_region)

        main_layout = QVBoxLayout()
        main_layout.addLayout(annotation_layout)
        main_layout.addLayout(video_layout)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_counter_label)
        main_layout.addLayout(slider_layout)
        main_layout.addLayout(control_layout)
        main_layout.addLayout(process_layout)
        self.setLayout(main_layout)

        # --- Signal connections ---
        self.btn_load.clicked.connect(self.load_video)
        self.model_combo.currentIndexChanged.connect(self.select_model)
        self.btn_load_sam.clicked.connect(self.load_sam_model)
        self.btn_process.clicked.connect(self.process_video)
        self.btn_save_seg.clicked.connect(self.save_segmentation)
        self.btn_load_seg.clicked.connect(self.load_segmentation)
        self.btn_prev_frame.clicked.connect(self.show_previous_frame)
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        self.btn_next_frame.clicked.connect(self.show_next_frame)
        self.btn_positive_annotation.clicked.connect(self.activate_positive_annotation)
        self.btn_negative_annotation.clicked.connect(self.activate_negative_annotation)
        self.vein_combo.currentIndexChanged.connect(self.select_vein)
        self.btn_clear_annotations.clicked.connect(self.clear_frame_annotations)
        self.btn_clear_all_annotations.clicked.connect(self.clear_all_annotations)
        self.btn_import_annotations.clicked.connect(self.import_annotations_from_file)
        self.left_image_label.clicked.connect(self.handle_click)
        self.left_image_label.scrolled.connect(self.handle_scroll)
        self.btn_crop_region.clicked.connect(self.toggle_crop_mode)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)

        # File watcher: auto-load data/auto_annotations.json when it changes on disk
        self._auto_ann_path = os.path.abspath(os.path.join("data", "auto_annotations.json"))
        self._file_watcher = QFileSystemWatcher()
        if os.path.exists(self._auto_ann_path):
            self._file_watcher.addPath(self._auto_ann_path)
        # Watch the parent directory so we also catch file creation
        _watch_dir = os.path.dirname(self._auto_ann_path)
        os.makedirs(_watch_dir, exist_ok=True)
        self._file_watcher.addPath(_watch_dir)
        self._file_watcher.fileChanged.connect(self._on_auto_annotation_file_changed)
        self._file_watcher.directoryChanged.connect(self._on_watch_dir_changed)


    def load_video(self):
        """Open a file dialog to select and load a video file."""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if video_path:
            self.video_path = video_path
            self.cap = cv2.VideoCapture(video_path)
            self.playing = False
            self.timer.stop()
            self.frames = []
            self.segmented_frames = []
            self.current_frame_idx = 0
            self.displayed_frame_idx = 0
            self.btn_play_pause.setText("Play")
            self.btn_play_pause.setEnabled(False)
            self.btn_prev_frame.setEnabled(False)
            self.btn_next_frame.setEnabled(False)
            self.frame_counter_label.setText("0 / 0")

            # Reset annotations
            self.clear_all_annotations()

            # Read all frames from the video
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frames.append(frame.copy())
            self.cap.release()

            # Display the first frame and set default crop region
            if self.frames:
                h, w = self.frames[0].shape[:2]
                # Hard coded crop region initially (relative fractions converted to px)
                # self.crop_region = [int(0.2125 * w), int(0.1125 * h), int(0.6792 * w), int(0.8479 * h)] # [xmin, ymin, xmax, ymax]
                self.left_image_label.set_crop_region(self.crop_region)
                self.right_image_label.clear()
                self.right_image_label.pixmap_image = None
                self.frame_slider.blockSignals(True)
                self.frame_slider.setEnabled(True)
                self.frame_slider.setRange(0, len(self.frames) - 1)
                self.frame_slider.setValue(0)
                self.frame_slider.blockSignals(False)
                self.btn_play_pause.setEnabled(True)
                total_frames = len(self.frames)
                self.frame_counter_label.setText(f"1 / {total_frames}")
                self.display_frame(0)
            else:
                self.frame_slider.blockSignals(True)
                self.frame_slider.setEnabled(False)
                self.frame_slider.setRange(0, 0)
                self.frame_slider.setValue(0)
                self.frame_slider.blockSignals(False)
                self.right_image_label.clear()
                self.right_image_label.pixmap_image = None
                self.btn_prev_frame.setEnabled(False)
                self.btn_next_frame.setEnabled(False)
                self.frame_counter_label.setText("0 / 0")
        self.set_active_label()
        video_name = os.path.basename(video_path) 
        self.setWindowTitle(f"Ultrasound Vein Segmentation - Prototype ({video_name})")

    def select_model(self):
        """Update the selected SAM model."""
        model_enum = self.model_combo.currentData()
        self.sam_model = SamModel(model_enum)  # store chosen enum value
        print(f"Selected SAM model: {model_enum.name}")

    def load_sam_model(self):
        """Load and initialize the SAM model for segmentation."""
        if not self.video_path:
            return
        
        if self.segmenter is not None:
            print("clearing the old sam model")
            try:
                # Let the segmenter clean itself up first
                self.segmenter.reset()
            except Exception as e:
                print(f"segmenter reset skipped: {e}")

            try:
                # Manually clear heavy objects if still present
                if hasattr(self.segmenter, "predictor"):
                    del self.segmenter.predictor
            except Exception as e:
                print(f"no predictor found in current segmenter: {e}")

            # Release GPU memory
            del self.segmenter
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        segmenter = SamVideoSegmenter(self.video_path, self.sam_model, n_frames=None)
        segmenter.extract_video_frames()     # ensure segmenter has frames loaded
        segmenter.initialize_inference()     # prepare model / device
        self.segmenter = segmenter
        print("SAM model loaded and initialized.")

    def process_video(self):
        """Run segmentation on the loaded video using the SAM model."""
        if not self.segmenter:
            return

        # Reset the segmenter memory
        self.segmenter.reset()

        # Add all annotated points to the segmenter
        for obj_id, prompt in self.annotations.items():
            has_frame0 = False
            for frame_idx in self.annotated_frames:
                mask_selection = np.array(prompt["frames"]) == frame_idx
                selected_points = np.array(prompt["points"])[mask_selection]
                selected_labels = np.array(prompt["labels"])[mask_selection]
                if len(selected_points) > 0:
                    self.segmenter.add_points(frame_idx=frame_idx, obj_id=obj_id,
                                             points=selected_points, labels=selected_labels)
                    if frame_idx == 0:
                        has_frame0 = True
            if not has_frame0:
                # Add hidden point for frame 0 if no annotation exists for frame 0
                # Allows to have multiple veins appearing at different times
                print(f"Adding hidden points for the first frame for vein {obj_id}")
                self.segmenter.add_points(frame_idx=0, obj_id=obj_id,
                                         points=np.array([[0, 0]]), labels=np.array([0]))

        # Propagate segmentation through all frames
        self.segmented_frames = self.full_segmentation()

        # Display the actual
        if self.segmented_frames:
            self.display_frame(self.displayed_frame_idx)

    def full_segmentation(self):
        """Segment all frames and return the processed frames."""
        self.segmenter.propagate()  # run propagation / tracking across frames
        processed_frames = []
        for frame_idx in range(len(self.frames)):
            masked = self.segmenter.get_masked_frame(frame_idx)
            if masked is None:
                processed_frames.append(self.frames[frame_idx])
            else:
                processed_frames.append(masked)

        # Auto-save immediately after every propagation run. Previously a run's
        # results only existed in memory until someone clicked "Save Segmentation" —
        # loading a different video (or a crash) before that click silently lost the
        # whole segmentation with no way to recover it. Auto-saving here means a
        # finished run is never lost, and each subsequent run/save overwrites the
        # same .npz for this video (see save_segmentations' atomic replace).
        try:
            self.segmenter.save_segmentations(crop_region=self.crop_region)
            print("[full_segmentation] Auto-saved segmentation after propagation.")
        except Exception as e:
            QMessageBox.warning(
                self, "Auto-save Failed",
                f"Propagation finished but auto-saving the segmentation failed:\n\n{e}\n\n"
                f"Your results are still in memory — use 'Save Segmentation' to retry "
                f"before loading a different video, or they will be lost."
            )
            print(f"[full_segmentation] AUTO-SAVE FAILED: {e}")

        return processed_frames

    def save_segmentation(self):
        """Save the segmented video frames."""
        if not self.segmented_frames:
            return
        # delegate saving to the segmenter (supports crop_region)
        self.segmenter.save_segmentations(crop_region=self.crop_region)
        print("Saving completed.")

    def load_segmentation(self):
        """Load a previously saved .npz segmentation and display it as an overlay,
        without re-running SAM. Frame indices in the .npz must match the currently
        loaded video's frame numbering — if you trimmed/re-cut the video since the
        .npz was saved, remap it first (see qc_flags.py / the frame-index remap done
        when a video gets cleaned up) or the masks will land on the wrong frames.
        """
        if not self.segmenter:
            print("Load a video and click 'Load SAM' first — the segmenter needs to "
                  "know the frame size/paths before masks can be displayed.")
            return

        npz_path, _ = QFileDialog.getOpenFileName(
            self, "Select Segmentation .npz", "", "NumPy Archive (*.npz)"
        )
        if not npz_path:
            return

        # np.load() on a .npz keeps the zip file handle open for lazy per-array access.
        # If we never close it, that handle stays alive for the life of this object
        # (which on Windows blocks any later rename/overwrite of this exact file —
        # this is what caused the earlier "Device or resource busy" bug). Load eagerly
        # inside a `with` block and close immediately instead of holding it open.
        with np.load(npz_path) as data:
            npz_files = list(data.files)
            npz_arrays = {key: data[key] for key in npz_files}

        # Guard against the exact mismatch that silently misaligns masks: loading a
        # .npz whose frame count/indices don't match the currently loaded video (e.g.
        # picking a pre-cut .npz for a post-cut video, or vice versa).
        npz_frame_indices = {int(k) for k in npz_files}
        video_frame_indices = set(range(len(self.frames)))
        if npz_frame_indices and not npz_frame_indices.issubset(video_frame_indices):
            max_npz_idx = max(npz_frame_indices)
            QMessageBox.warning(
                self, "Frame Count Mismatch",
                f"This .npz has frame indices up to {max_npz_idx}, but the loaded "
                f"video only has {len(self.frames)} frames (0-{len(self.frames)-1}).\n\n"
                f"This almost always means the .npz was saved for a DIFFERENT video "
                f"(e.g. before/after frames were cut) and will misalign masks to the "
                f"wrong frames if loaded. Not loading it — pick the .npz that matches "
                f"this exact video."
            )
            print(f"[load_segmentation] REFUSED: npz max frame idx {max_npz_idx} "
                  f"exceeds video's {len(self.frames)} frames — likely wrong file.")
            return
        if len(npz_frame_indices) < len(video_frame_indices) * 0.5:
            print(f"[load_segmentation] WARNING: npz only covers {len(npz_frame_indices)} "
                  f"of this video's {len(self.frames)} frames — may be a partial/mismatched file.")

        video_segments = {}
        obj_ids = set()
        for key in npz_files:
            frame_idx = int(key)
            arr = npz_arrays[key]  # [num_obj, H, W]
            video_segments[frame_idx] = {}
            for obj_id in range(arr.shape[0]):
                video_segments[frame_idx][obj_id] = arr[obj_id]
                obj_ids.add(obj_id)

        self.segmenter.video_segments = video_segments
        self.segmenter.obj_ids = obj_ids

        self.segmented_frames = []
        for frame_idx in range(len(self.frames)):
            masked = self.segmenter.get_masked_frame(frame_idx)
            self.segmented_frames.append(masked if masked is not None else self.frames[frame_idx])

        print(f"Loaded segmentation for {len(video_segments)} frames from {npz_path} "
              f"({len(obj_ids)} object(s)) — displaying, no propagation run.")
        self.display_frame(self.displayed_frame_idx)

    def start_video(self):
        """Start video playback."""
        if self.frames:
            if self.current_frame_idx >= len(self.frames):
                self.current_frame_idx = 0
                self.displayed_frame_idx = 0
            if not self.playing:
                self.playing = True
                self.timer.start(int(1000 * (1 / self.fps)))  # ms per frame
                self.btn_play_pause.setText("Pause")

    def pause_video(self):
        """Pause video playback."""
        self.playing = False
        self.timer.stop()
        self.btn_play_pause.setText("Play")

    def update_fps(self):
        """Update the playback FPS from the input field."""
        try:
            self.fps = int(self.fps_input.text())
        except ValueError:
            self.fps = 60
            self.fps_input.setText(str(self.fps))

    def update_frame(self):
        """Update the displayed frames during playback."""
        if self.frames and self.playing:
            if self.current_frame_idx >= len(self.frames):
                self.timer.stop()
                self.playing = False
                self.btn_play_pause.setText("Play")
                return
            self.display_frame(self.current_frame_idx)
            self.current_frame_idx += 1

    def show_previous_frame(self):
        """Display the previous frame when available."""
        if not self.frames:
            return
        if self.playing:
            self.pause_video()
        target_idx = max(0, self.displayed_frame_idx - 1)
        self.display_frame(target_idx)

    def show_next_frame(self):
        """Display the next frame when available."""
        if not self.frames:
            return
        if self.playing:
            self.pause_video()
        target_idx = min(len(self.frames) - 1, self.displayed_frame_idx + 1)
        self.display_frame(target_idx)

    def toggle_play_pause(self):
        """Toggle the playback state for the video."""
        if not self.frames:
            return
        if self.playing:
            self.pause_video()
        else:
            self.start_video()
    
    def activate_positive_annotation(self):
        """Activate positive annotation mode (foreground)."""
        self.btn_positive_annotation.setChecked(True)
        self.btn_negative_annotation.setChecked(False)
        self.set_active_label()
    
    def activate_negative_annotation(self):
        """Activate negative annotation mode (background)."""
        self.btn_positive_annotation.setChecked(False)
        self.btn_negative_annotation.setChecked(True)
        self.set_active_label()

    def select_vein(self):
        """Set the current vein id for annotation."""
        vein_enum = self.vein_combo.currentData()
        self.current_vein = Vein(vein_enum)
        print(f"Current vein: {self.current_vein.label}")
        self.set_active_label()
    
    def set_active_label(self):
        current_vein_id = self.current_vein.id
        is_positive = self.btn_positive_annotation.isChecked()
        self.left_image_label.set_active_label(current_vein_id, is_positive)

    def clear_frame_annotations(self):
        """Clear annotation points and labels for current_frame_idx."""
        empty_labels = []
        for label, annotation in self.annotations.items():
            points = np.array(annotation['points'])
            labels = np.array(annotation['labels'])
            frames = np.array(annotation['frames'])
            mask = ~(frames == self.current_frame_idx)
            annotation['points'] = points[mask].tolist()
            annotation['labels'] = labels[mask].tolist()
            annotation['frames'] = frames[mask].tolist()
            if len(annotation['frames']) == 0:
                empty_labels.append(label)
        for label in empty_labels:
            self.annotations.pop(label)
        self.annotated_frames.discard(self.current_frame_idx)
        self.left_image_label.clear_annotation_marks()
        self.update()
    
    def clear_all_annotations(self):
        """Clears ALL annotation points and labels if no frame_idx provided."""
        self.annotations.clear()
        self.annotated_frames.clear()    
        self.left_image_label.clear_annotation_marks()
        self.update()

    # ── Auto-annotation import ───────────────────────────────────────────────

    def inject_annotations(self, data: dict):
        """
        Programmatically load annotations into the app from a dict.
        Expected format (same as data/auto_annotations.json):
          {
            "annotations": {
              "0": {"points": [[x,y],...], "labels": [1,0,...], "frames": [idx,...]},
              "1": {...}
            },
            "annotated_frames": [0, 5, 10]
          }
        Merges with any existing annotations (clear first if you want a clean slate).
        """
        raw = data.get("annotations", {})
        for vein_id_str, ann in raw.items():
            vein_id = int(vein_id_str)
            self.annotations.setdefault(vein_id, {"points": [], "labels": [], "frames": []})
            self.annotations[vein_id]["points"].extend(ann.get("points", []))
            self.annotations[vein_id]["labels"].extend(ann.get("labels", []))
            self.annotations[vein_id]["frames"].extend(ann.get("frames", []))

        for idx in data.get("annotated_frames", []):
            self.annotated_frames.add(int(idx))

        # Rebuild the annotation marks for the currently displayed frame
        self.display_frame(self.displayed_frame_idx)
        n_pts = sum(len(v["points"]) for v in self.annotations.values())
        print(f"[inject_annotations] Loaded {n_pts} annotation points across "
              f"{len(self.annotated_frames)} frame(s).")

    def import_annotations_from_file(self, filepath: str = None):
        """Load annotations from a JSON file. Opens a dialog if filepath is None."""
        if not filepath:
            filepath, _ = QFileDialog.getOpenFileName(
                self, "Import Annotations", "data", "JSON Files (*.json)"
            )
        if not filepath or not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            self.inject_annotations(data)
            print(f"[import_annotations] Loaded from {filepath}")
        except Exception as e:
            QMessageBox.warning(self, "Import Error", f"Failed to load annotations:\n{e}")

    def _on_auto_annotation_file_changed(self, path: str):
        """Called by QFileSystemWatcher when data/auto_annotations.json is modified."""
        if not os.path.exists(path):
            return
        print(f"[file watcher] Detected change: {path}  — auto-loading annotations…")
        # Re-add path because some editors replace the file (unwatch + rewatch)
        if path not in self._file_watcher.files():
            self._file_watcher.addPath(path)
        self.import_annotations_from_file(path)

    def _on_watch_dir_changed(self, directory: str):
        """Called when the watched directory changes (catches file creation)."""
        if os.path.exists(self._auto_ann_path):
            if self._auto_ann_path not in self._file_watcher.files():
                self._file_watcher.addPath(self._auto_ann_path)
                print(f"[file watcher] Now watching {self._auto_ann_path}")
                self.import_annotations_from_file(self._auto_ann_path)

    # ── Crop / scroll / click ────────────────────────────────────────────────

    def toggle_crop_mode(self):
        self.crop_mode = self.btn_crop_region.isChecked()
        if not self.crop_mode:
            self.crop_first_point = None
        self.left_image_label.set_crop_mode(self.crop_mode)

    def handle_scroll(self, is_up, event):
        current_idx = self.vein_combo.currentIndex()
        if is_up:
            new_idx = current_idx - 1
        else:
            new_idx = current_idx + 1
        final_idx = max(0, min(new_idx, self.vein_combo.count() - 1))
        self.vein_combo.setCurrentIndex(final_idx)
        self.select_vein()
    
    def handle_click(self, x, y, event):
        """
        Handle click events for annotation or crop region.
        """
        print(f"[UltrasoundApp] Received click at ({x}, {y})")
        
        if event.button() == Qt.LeftButton:
            self.activate_positive_annotation()
        elif event.button() == Qt.RightButton:
            self.activate_negative_annotation()

        # If in crop mode, handle crop region selection
        if self.crop_mode:
            if self.crop_first_point is None:
                self.crop_first_point = (x, y)
                self.left_image_label.set_temp_crop(self.crop_first_point, None)
                print(f"Crop first point set at: {self.crop_first_point}")
            else:
                x0, y0 = self.crop_first_point
                x1, y1 = x, y
                xmin, xmax = sorted([x0, x1])
                ymin, ymax = sorted([y0, y1])
                self.crop_region = [xmin, ymin, xmax, ymax]
                self.left_image_label.set_crop_region(self.crop_region)
                print(f"Crop region set to: {self.crop_region}")
                self.crop_first_point = None
                self.left_image_label.set_temp_crop(None, None)
                self.btn_crop_region.setChecked(False)
                self.crop_mode = False
                self.left_image_label.set_crop_mode(False)
                print("Crop mode deactivated.")

        # Otherwise, handle annotation point
        else:
            # Ensure the current vein id has an entry in the annotations dictionary
            current_vein_id = self.current_vein.id
            self.annotations.setdefault(current_vein_id, {"points": [], "labels": [], "frames": []})
            # Determine label: 1 for positive presence, 0 for negative presence
            is_positive = self.btn_positive_annotation.isChecked()
            label_value = 1 if is_positive else 0
            # Add the point and label to the current vein id annotation
            self.annotations[current_vein_id]["points"].append([x, y])
            self.annotations[current_vein_id]["labels"].append(label_value)
            self.annotations[current_vein_id]["frames"].append(self.displayed_frame_idx)
            # Mark this frame as having annotations
            self.annotated_frames.add(self.displayed_frame_idx)
            print(f"Current annotated frames: {self.annotated_frames}")
            # Add annotation mark to the label for drawing
            self.left_image_label.add_annotation_mark(x, y, current_vein_id, is_positive)
            print(f"Current annotations: {self.annotations}")

    def display_frame(self, frame_idx):
        """Display a specific frame and synchronize UI state."""
        if not self.frames:
            return
        clamped_idx = max(0, min(frame_idx, len(self.frames) - 1))
        frame = self.frames[clamped_idx]
        qt_original = cv_to_qt_pixmap(frame)
        self.left_image_label.setPixmap(qt_original)
        self.left_image_label.pixmap_image = qt_original

        # If there are annotations for this frame, update the left label's marks
        if clamped_idx in self.annotated_frames:
            marks = []
            for instance, ann in self.annotations.items():
                for point, label, frame_idx in zip(ann["points"], ann["labels"], ann["frames"]):
                    if frame_idx == clamped_idx:
                        marks.append((point[0], point[1], instance, label == 1))
            self.left_image_label.annotation_marks = marks
            self.left_image_label.update()
        else:
            self.left_image_label.clear_annotation_marks()

        # Show segmented/processed frame on the right if available
        if self.segmented_frames and clamped_idx < len(self.segmented_frames):
            processed_colored = self.segmented_frames[clamped_idx]
            qt_processed = cv_to_qt_pixmap(processed_colored)
            self.right_image_label.setPixmap(qt_processed)
            self.right_image_label.pixmap_image = qt_processed
        elif self.right_image_label.pixmap() is not None:
            self.right_image_label.clear()
            self.right_image_label.pixmap_image = None

        # Sync indices and UI controls
        self.displayed_frame_idx = clamped_idx
        self.current_frame_idx = clamped_idx

        if self.frames:
            self.btn_prev_frame.setEnabled(clamped_idx > 0)
            self.btn_next_frame.setEnabled(clamped_idx < len(self.frames) - 1)
            self.frame_counter_label.setText(f"{clamped_idx + 1} / {len(self.frames)}")

        if self.frame_slider.isEnabled():
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(clamped_idx)
            self.frame_slider.blockSignals(False)

    def on_frame_slider_changed(self, value):
        """Handle manual frame selection via slider."""
        if self.playing:
            self.pause_video()
        self.display_frame(value)