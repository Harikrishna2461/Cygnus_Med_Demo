"""
Stage 2: Human Pose Estimation

Takes in RGB frame and does human pose estimation using the bounding boxes.
Keypoints and scores are arranged into COCO human pose annotation format with
shape (2, 17, 2), arranged as x,y-coordinates of 17 keypoints, first for the
patient and second for the doctor.

The results are updated to the PositionMemory.
"""
import threading
import traceback
import queue
import numpy as np

from .position_memory import PositionMemory
from ..model_paths.model_loader import load_pose_model


class PoseDetector:
    def __init__(self, position_memory: PositionMemory, barrier, exit_event, device='cuda', backend='onnxruntime'):
        """
        Args:
            position_memory (PositionMemory): PositionMemory to read/update algorithm results.
            barrier (threading.Barrier): Barrier to stop the loop until other stages are done.
            exit_event (threading.Event): Control for thread to keep running or exit.
            device (str, optional): Device to load inference model. Defaults to 'cuda'.
            backend (str, optional): Backend framework for inference model. Defaults to 'onnxruntime'.
        """
        self.input_queue = queue.Queue(maxsize=1)
        self.position_memory = position_memory
        self.barrier = barrier
        self.exit_event = exit_event
        self.pose_model = load_pose_model(filename='rtmpose-l-192.onnx',
                                          input_size=(192,256),
                                          backend=backend,
                                          device=device)
        self.worker_thread = threading.Thread(target=self.run, daemon=True)
    
    def start(self):
        self.worker_thread.start()
        print("    PoseDetector thread started.")
    
    def stop(self):
        self.exit_event.set()
        self.worker_thread.join()
        
    def run(self):
        while not self.exit_event.is_set():
            try:
                try:
                    rgb_frame = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue
                # Run pose estimation using found bboxes.
                has_found_bbox, bboxes = self.position_memory.get_bboxes()
                if has_found_bbox:
                    keypoints, scores, swap_flag = self.estimate_pose(rgb_frame, bboxes)
                    # Note: Repeated poor results reset bboxes.
                    self.position_memory.update_keypoints(keypoints, scores, swap_flag)
                self.barrier.wait()
            
            except threading.BrokenBarrierError:
                break
            except Exception:
                print(f"ERROR in PoseDetector thread:")
                traceback.print_exc()
                self.exit_event.set()
                self.barrier.abort()
                break
        print("    PoseDetector thread closed.")

    def estimate_pose(self, rgb_frame, bboxes):
        if bboxes is None:
            return None, None, None
        
        keypoints, scores = self.pose_model(rgb_frame, bboxes)
        try:
            p_kp, p_sc, d_kp, d_sc, swap_flag = self.identify_person(keypoints, scores)
        except ValueError as e:
            raise RuntimeError("Model returned invalid output format") from e
        except Exception:
            raise
        
        if p_kp is None:
            return None, None, None
        
        # Pack it as keypoints and scores again.
        keypoints = np.asarray([p_kp, d_kp])
        scores = np.asarray([p_sc, d_sc])
        return keypoints, scores, swap_flag
    
    def identify_person(self, keypoints, scores, thr=0.5):
        """
        Identifies (differentiates) patient and doctor from the pose and returns their keypoints.
        
        Args:
            keypoints, scores: Returned pose detection from RTMLib, use directly.
            thr: Threshold to consider a body part as successfully detected, default 0.5.
        
        Returns:
            patient_keypoints: Patient's keypoints (17 COCO keypoints), None if no/bad detection.
            patient_scores: Scores corresponding to patinet's keypoints, None if no/bad detection.
            doctor_keypoints: Doctor's keypoints (17 COCO keypoints), None if no/bad detection.
            doctor_scores: Scores corresponding to doctor's keypoints, None if no/bad detection.
            
        Throws:
            ValueError: When model output is in wrong format, e.g. not 17 keypoints or
                        mismatch in number of keypoints and scores.
        """
        if keypoints.shape[0] != scores.shape[0]:
            raise ValueError(
                f"Invalid model output: "
                f"{len(keypoints)} instances but {len(scores)} scores."
            )
        
        if keypoints.shape[1] != 17 or scores.shape[1] != 17:
            raise ValueError(
                f"Invalid model output: expected 17 points but detected"
                f"{keypoints.shape[1]} keypoints and {scores.shape[1]} scores."
            )
        
        # Return empty keypoints if there are more/less than 2 targets in the frame.
        if len(keypoints) != 2:
            # print("num targets:", len(keypoints))
            return None, None, None, None, False
        
        # Find existing arms to identify the scanner hand.
        patient_keypoints, patient_scores, doctor_keypoints, doctor_scores = None, None, None, None
        first = None
        swap_bboxes = False
        if scores[0][5] > thr or scores[0][6] > thr:  # first instance has shoulders
            patient_keypoints, patient_scores = keypoints[1], scores[1]
            doctor_keypoints, doctor_scores = keypoints[0], scores[0]
            first = keypoints[0][5] if scores[0][5] > scores[0][6] else keypoints[0][6]
            swap_bboxes = True
        if scores[1][5] > thr or scores[1][6] > thr:  # second instance has shoulders
            second = keypoints[1][5] if scores[1][5] > scores[1][6] else keypoints[1][6]
            if first is None or first[1] < second[1]:
                patient_keypoints, patient_scores = keypoints[0], scores[0]
                doctor_keypoints, doctor_scores = keypoints[1], scores[1]
                swap_bboxes = False
        
        return patient_keypoints, patient_scores, doctor_keypoints, doctor_scores, swap_bboxes