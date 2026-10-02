"""
Stage 1: Detect bounding box of humans

Takes in RGB frame, if a bounding box needs to be found then it runs a
light-weight inference model to identify two bounding boxes in the frame.

Two highest confident bounding boxes are returned. Returns None if there are
less than two bounding boxes.

The results are updated to the PositionMemory.
"""

import threading
import traceback
import queue
import numpy as np

from .position_memory import PositionMemory
from ..model_paths.model_loader import load_det_model
from .utils import calculate_iou


class PersonDetector:
    """
    Wrapper for thread detecting bounding boxes of people.
    Pass RGB frames for processing into `input_queue`.
    """
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
        self.det_model = load_det_model(filename='rtmdet-m-640.onnx',
                                        input_size=(320,320),
                                        backend=backend,
                                        device=device)
        self.worker_thread = threading.Thread(target=self.run, daemon=True)
    
    def start(self):
        self.worker_thread.start()
        print("    PersonDetector thread started.")
    
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
                # Get bounding boxes for patient and doctor for top-down inference.
                has_found_bbox, bboxes = self.position_memory.get_bboxes()
                if not has_found_bbox:
                    pose_blackbox, seg_blackbox = self.position_memory.get_bbox_blacklist()
                    found_bboxes, found_scores = self.find_bboxes(rgb_frame, bboxes, pose_blackbox, seg_blackbox)
                    if len(found_bboxes) == 2:  # updat only if 2 boxes
                        self.position_memory.update_bboxes(found_bboxes)
                    elif len(found_bboxes) < 2:
                        self.position_memory.reset_bbox_blacklist()
                self.barrier.wait()
                
            except threading.BrokenBarrierError:
                break
            except Exception:
                print(f"ERROR in PersonDetector thread:")
                traceback.print_exc()
                self.exit_event.set()
                self.barrier.abort()
                break
        print("    PersonDetector thread closed.")
    
    def find_bboxes(self, rgb_frame, bboxes, pose_blackbox, seg_blackbox):
        found_bboxes, found_scores = [], []
        det_bboxes, det_scores = self.det_model(rgb_frame)
        
        # Remove blacklisted bboxes.
        for det_bbox, det_score in zip(det_bboxes, det_scores):
            is_blacklist = False
            if len(seg_blackbox) > 0:
                bbox_blacklist = seg_blackbox
            else:
                bbox_blacklist = pose_blackbox
            for bad_bbox in bbox_blacklist:
                if calculate_iou(bad_bbox, det_bbox) > 0.5:
                    is_blacklist = True
                    break
            if not is_blacklist:
                found_bboxes.append(det_bbox)
                found_scores.append(det_score)
        found_bboxes = np.asarray(found_bboxes)
        found_scores = np.asarray(found_scores)
        # Base case: Initial run with no memory.
        if bboxes is None or len(found_bboxes) < 2:
            # Use the two bounding boxes with highest scores.
            sorted_scores = np.argsort(found_scores)
            # print(sorted_scores)
            found_bboxes = found_bboxes[sorted_scores[-2:]]
            found_scores = found_scores[sorted_scores[-2:]]
            return found_bboxes, found_scores
        
        # Use previous bboxes for IoU.
        iou_bbox1 = []
        iou_bbox2 = []
        for bbox in found_bboxes:
            iou_bbox1.append(calculate_iou(bbox, bboxes[0]))
            iou_bbox2.append(calculate_iou(bbox, bboxes[1]))
        iou_argsort_1 = np.argsort(iou_bbox1)
        iou_argsort_2 = np.argsort(iou_bbox2)
        max_iou_1 = iou_argsort_1[-1]
        max_iou_2 = iou_argsort_2[-1]
        
        # Check for any overlaps and do tiebreaker.
        if max_iou_1 != max_iou_2:
            bbox1 = found_bboxes[max_iou_1]
            score1 = found_scores[max_iou_1]
            bbox2 = found_bboxes[max_iou_2]
            score2 = found_scores[max_iou_2]
            return [bbox1, bbox2], [score1, score2]
        
        # Tie-breaker.
        if iou_bbox1[max_iou_1] > iou_bbox2[max_iou_2]:
            max_iou_2 = iou_argsort_2[-2]
            bbox1 = found_bboxes[max_iou_1]
            score1 = found_scores[max_iou_1]
            bbox2 = found_bboxes[max_iou_2]
            score2 = found_scores[max_iou_2]
            return [bbox1, bbox2], [score1, score2]
        else:
            max_iou_1 = iou_argsort_1[-2]
            bbox1 = found_bboxes[max_iou_1]
            score1 = found_scores[max_iou_1]
            bbox2 = found_bboxes[max_iou_2]
            score2 = found_scores[max_iou_2]
            return [bbox1, bbox2], [score1, score2]