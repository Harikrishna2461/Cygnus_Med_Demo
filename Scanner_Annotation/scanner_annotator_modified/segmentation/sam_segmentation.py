#%%
import json
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import urllib.request
# from sam_segmentation_utils import extract_frames
from sam2.build_sam import build_sam2_video_predictor
from segmentation.sam_models import SamModel
from tqdm import tqdm

from segmentation.veins import Vein


class SamVideoSegmenter:
    def __init__(self, video_path, model_size=SamModel.BASE_PLUS, n_frames=None, device=None,
                 max_cached_frames=512):
        self.video_path = video_path
        self.model_size = model_size
        self.n_frames = n_frames
        # Bound how many decoded frames stay resident in memory at once instead of
        # preloading the whole clip as one tensor. At ~1024x1024x3 float32/frame
        # (~12MB), 512 frames caps out around ~6GB regardless of how long the merged
        # clip is. For clips shorter than this, behavior is identical to eager
        # preloading (nothing gets evicted). Set to None to restore the old
        # preload-everything behavior.
        self.max_cached_frames = max_cached_frames
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f"using device: {self.device}")

        self.sam2_checkpoint = f"./segmentation/checkpoints/{self.model_size.value[0]}.pt"
        self.model_cfg = f"./configs/sam2.1/{self.model_size.value[1]}.yaml"
        self.predictor = self.build_predictor()
        # lf.predictor.model.half() # use FP16 to accelarate
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.frame_names = []
        self.inference_state = None
        self.obj_ids = set()
        self.video_segments = {}
        self.crop_region = None

        self.data_dir = './data'
        self.frames_dir = None

    def download_model(self):
        """Download the SAM model checkpoint if not present."""
        if not os.path.exists(self.sam2_checkpoint):
            print(f"Downloading SAM model checkpoint to {self.sam2_checkpoint}...")
            checkpoint_url = self.model_size.value[2]

            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(self.sam2_checkpoint)) as t:
                urllib.request.urlretrieve(checkpoint_url, self.sam2_checkpoint, reporthook=t.update_to)

    def build_predictor(self):
        """Download the SAM model checkpoint if not present and build the video predictor."""
        # Force BODY and TAIL to be mutually exclusive at every pixel, at the model level,
        # instead of letting them overlap (yellow) and fixing it up after the fact:
        #   - non_overlap_masks: at output time, keep only the higher-confidence object per
        #     pixel (argmax across objects) - see SAM2Base._apply_non_overlapping_constraints.
        #   - non_overlap_masks_for_mem_enc: apply that same constraint to what gets written
        #     into the memory bank, so future/similar frames condition on a clean, already
        #     de-overlapped mask instead of an ambiguous one - this is what should make
        #     interpolation across similar frames consistent rather than jittery.
        non_overlap_overrides = [
            "++model.non_overlap_masks=true",
            "++model.non_overlap_masks_for_mem_enc=true",
        ]

        if self.model_size == SamModel.EDGETAM:
            model_path = self.model_size.value[2]
            device = self.device
            config_file = "./configs/sam2.1/edgetam.yaml"

            predictor = build_sam2_video_predictor(
                config_file=config_file, ckpt_path=model_path, device=device,
                hydra_overrides_extra=non_overlap_overrides,
            )
            return predictor

        else:
            self.download_model()
            return build_sam2_video_predictor(
                self.model_cfg, self.sam2_checkpoint, device=self.device,
                hydra_overrides_extra=non_overlap_overrides,
            )

    def extract_video_frames(self):
        """Extract frames from the video and store them in a directory."""
        # Create output directory based on video name
        global_frames_dir = os.path.join(self.data_dir, "frames")
        output_dir = os.path.join(global_frames_dir, self.video_name)
        self.frames_dir = output_dir

        # Ensure frames are extracted and frames_dir is set and non-empty (contains image frames)
        need_extract = False
        if self.frames_dir is None or not os.path.isdir(self.frames_dir):
            need_extract = True
        else:
            imgs = [
            p for p in os.listdir(self.frames_dir)
            if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png")
            ]
            if len(imgs) == 0:
                need_extract = True

        # Extract only if directory does not exist
        if need_extract:
            os.makedirs(output_dir, exist_ok=True)

            # Load the video
            cap = cv2.VideoCapture(self.video_path)

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_name = f"{frame_idx:05d}.jpg"
                frame_path = os.path.join(output_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                frame_idx += 1

            cap.release()
            print(f"Extracted {frame_idx} frames to {output_dir}")
            print(f"Directory {output_dir} already exists. Skipping extraction.")
        
        self.frame_names = [
            p for p in os.listdir(output_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    def initialize_inference(self):
        """Initialize the inference state for video segmentation."""
        if self.frames_dir is None:  # Try extracting frames for the first time
            self.extract_video_frames()
        self.inference_state = self.predictor.init_state(
            video_path=self.frames_dir,
            max_cached_frames=self.max_cached_frames,
        )

    def add_points(self, frame_idx, obj_id, points, labels):
        """Add point prompts for a specific frame and object ID."""
        if self.inference_state is None:
            self.initialize_inference()
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        self.obj_ids.add(obj_id)

    # def propagate(self):
    #     """Propagate the masks through the video frames."""
    #     if self.inference_state is None:  # Initialize inference if not done
    #         self.initialize_inference()
    #     self.video_segments = {}
    #     with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
    #         for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
    #             self.video_segments[out_frame_idx] = {
    #                 out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
    #                 for i, out_obj_id in enumerate(out_obj_ids)
    #             }

    def propagate(self, batch_size=4):
        """Propagate the masks through the video frames using batch GPU inference."""
        if self.inference_state is None:  # Initialize inference if not done
            self.initialize_inference()

        self.video_segments = {}

        try:
            import torch
            if hasattr(torch, "compile"):
                self.predictor = torch.compile(self.predictor)
                print("[INFO] Predictor compiled with torch.compile for speed optimization.")
        except Exception as e:
            print(f"[WARN] Torch compile skipped: {e}")

        batch_frames = []
        batch_frame_indices = []
        
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(self.inference_state):
                batch_frames.append(mask_logits)  # Assume mask_logits is a tensor
                batch_frame_indices.append((frame_idx, obj_ids))
                
                # When batch is full, process it on GPU
                if len(batch_frames) == batch_size:
                    batch_masks = torch.stack(batch_frames)  # [B, N, H, W]
                    
                    # Apply threshold and move to CPU for numpy conversion
                    batch_masks = (batch_masks > 0.0).cpu().numpy()
                    
                    # Store the segmentation results for each frame
                    for i, (idx, ids) in enumerate(batch_frame_indices):
                        self.video_segments[idx] = {
                            obj_id: batch_masks[i][j]
                            for j, obj_id in enumerate(ids)
                        }
                    
                    # Clear the batch
                    del batch_masks
                    batch_frames.clear()
                    batch_frame_indices.clear()
                    torch.cuda.empty_cache()  # free unused GPU memory

            # Process any remaining frames that didn't fill the last batch
            if len(batch_frames) > 0:
                batch_masks = torch.stack(batch_frames)
                batch_masks = (batch_masks > 0.0).cpu().numpy()
                for i, (idx, ids) in enumerate(batch_frame_indices):
                    self.video_segments[idx] = {
                        obj_id: batch_masks[i][j]
                        for j, obj_id in enumerate(ids)
                    }
                del batch_masks
                torch.cuda.empty_cache()
    
    def reset(self):
        """Reset the predictor state and clear stored segmentations."""
        self.predictor.reset_state(self.inference_state)
        self.video_segments = {}
    
    def get_frame(self, frame_idx) -> np.ndarray:
        """Get the original frame at the specified index."""
        if self.frames_dir is None or frame_idx >= len(self.frame_names):
            return None
        frame_path = os.path.join(self.frames_dir, self.frame_names[frame_idx])
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_masked_frame(self, frame_idx) -> np.ndarray:
        """Get the frame with overlaid segmentation masks."""
        if self.inference_state is None or frame_idx not in self.video_segments:
            return None
        frame = self.get_frame(frame_idx)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        masks = self.video_segments[frame_idx]

        for i, obj_id in enumerate(self.obj_ids):
            vein = Vein.from_id(obj_id)
            color = vein.color
            # color = color.astype(np.uint8)
            mask = masks[obj_id]
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            cv2.addWeighted(frame, 1.0, mask_image, 0.5, 0, frame)

        return frame
    
    def save_segmentations(self, crop_region=None):
        """Save segmentation masks to disk.
        If crop_region is provided, only save the cropped region of the masks.
        """
        if not self.video_segments:  # Check if nothing to save
            return
        
        # Prepare output directories
        segmentations_dir = os.path.join(self.data_dir, "segmentations")
        metadata_dir = os.path.join(self.data_dir, "metadata")
        os.makedirs(segmentations_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)

        max_obj_id = max(self.obj_ids)

        # Collect masks for all frames and save as a single .npz file
        all_masks = {}
        for frame_idx, masks in self.video_segments.items():
            # Ensure all object IDs are present for this frame
            for obj_id in range(max_obj_id + 1):
                if obj_id not in masks:
                    h, w = list(masks.values())[0].shape[-2:]
                    masks[obj_id] = np.zeros((1, h, w), dtype=bool)

            # Stack all masks
            concatenated_masks = np.stack([masks[obj_id][0] for obj_id in range(max_obj_id + 1)], axis=0)

            # Erase outside crop region to avoid confusion
            if crop_region is not None:
                x_min, y_min, x_max, y_max = crop_region
                h, w = concatenated_masks.shape[-2:]
                full_mask = np.zeros((h, w), dtype=bool)
                full_mask[y_min:y_max, x_min:x_max] = True
                for obj_id in range(max_obj_id + 1):
                    concatenated_masks[obj_id] = concatenated_masks[obj_id] & full_mask

            all_masks[f"{frame_idx}"] = concatenated_masks

        segmentation_file_path = os.path.join(segmentations_dir, f"{self.video_name}.npz")
        # Write to a temp file first, then atomically replace the target. This lets
        # future runs/saves overwrite an existing .npz for this video cleanly instead
        # of failing outright, and avoids ever leaving a half-written/corrupt file if
        # the process dies mid-save.
        # NOTE: np.savez_compressed() silently APPENDS ".npz" to any path that doesn't
        # already end in ".npz" (e.g. "foo.npz.tmp" gets written as "foo.npz.tmp.npz").
        # The temp name must already end in ".npz" or the later os.replace() looks for
        # the wrong filename and fails, leaving the real data stranded under the
        # numpy-mangled name. Use a ".tmp.npz" suffix so numpy leaves it alone.
        tmp_file_path = os.path.join(segmentations_dir, f"{self.video_name}.tmp.npz")
        try:
            np.savez_compressed(tmp_file_path, **all_masks)
            os.replace(tmp_file_path, segmentation_file_path)
        except PermissionError as e:
            # Most common cause on Windows: another handle to this exact .npz is still
            # open in this same process (e.g. a previous Load Segmentation call that
            # didn't close its np.load() file handle) or the file is open elsewhere
            # (Explorer preview, another program). Surface this clearly instead of
            # silently losing the segmentation.
            if os.path.exists(tmp_file_path):
                try:
                    os.remove(tmp_file_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"Could not overwrite {segmentation_file_path} — it appears to be "
                f"locked/open elsewhere (e.g. still open from a previous Load "
                f"Segmentation, or open in another program). Close whatever has it "
                f"open and save again. Original error: {e}"
            ) from e
        print(f"Segmentation masks saved to {segmentation_file_path}")

        metadata_file_path = os.path.join(metadata_dir, f"{self.video_name}.json")
        metadata = {
            "crop_region": crop_region,
            "num_frames": len(self.frame_names),
        }
        with open(metadata_file_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_file_path}")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)