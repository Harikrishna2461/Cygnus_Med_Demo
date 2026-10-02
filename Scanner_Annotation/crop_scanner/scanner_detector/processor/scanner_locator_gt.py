"""
Stage 3: Scanner segmentation & position calculation

Takes in RGB frame, use the doctor's pose to crop a potential region for the scanner
and run instance segmentation on it using a fine-tuned RF-DETR model.

The results are used to calculate the coordinate of the tip of the scanner head.

The results are updated to the PositionMemory.
"""
import threading
import traceback
import queue
import json

import cv2
import numpy as np
import time

from .position_memory import PositionMemory
from ..model_paths.model_loader import load_scanner_seg_model
from .utils import find_contours, contour_touches_contour
from ..shared import debug_q, test_q


# Constants for cropping window around scanner.
CROP_W, CROP_H = 168, 168  # match RF-DETR
CROP_OFFSET = 80

class ScannerLocatorGT:
    def __init__(self, position_memory: PositionMemory, barrier, exit_event, filename):
        """
        Args:
            position_memory (PositionMemory): PositionMemory to read/update algorithm results.
            barrier (threading.Barrier): Barrier to stop the loop until other stages are done.
            exit_event (threading.Event): Control for thread to keep running or exit.
        """
        self.input_queue = queue.Queue(maxsize=1)
        self.position_memory = position_memory
        self.barrier = barrier
        self.exit_event = exit_event
        self.worker_thread = threading.Thread(target=self.run, daemon=True)
        
        self.masks = self.load_mask(filename)
        self.offsets = self.load_offset(filename)
        
    def load_mask(self, filename):
        mask_file = f'scanner_segment_gt/{filename}_scanner.npz'
        masks = np.load(mask_file, allow_pickle=True)
        return masks
    
    def load_offset(self, filename):
        offset_file = f'scanner_segment_gt/{filename}_scanner.json'
        with open(offset_file, 'r', encoding='utf-8') as json_file:
            gt = json.load(json_file)
        return gt
    
    def start(self):
        self.worker_thread.start()
        print("    ScannerLocator thread started.")
    
    def stop(self):
        self.exit_event.set()
        self.worker_thread.join()
        
    def run(self):
        while not self.exit_event.is_set():
            try:
                try:
                    frame_id = test_q.get(timeout=1)
                    frame = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Load GT masks + offset and use it for results.
                keypoints, scores = self.position_memory.get_keypoints()
                d_kp, d_sc = keypoints[1], scores[1]
                scanner_pos = self.find_scanner_pos(d_kp, d_sc, frame_id)
                self.position_memory.update_scanner(scanner_pos)
                
                self.barrier.wait()
            
            except threading.BrokenBarrierError:
                break
            except Exception:
                print(f"ERROR in ScannerLocator thread:")
                traceback.print_exc()
                self.exit_event.set()
                self.barrier.abort()
                break
        print("    ScannerLocator thread closed.")
    
    def find_scanner_pos(self, doctor_kp, doctor_scores, frame_id, thr=0.5):
        """
        Identify and returns the edge pixel of the white scanner.
        
        Args:
            rgb_frame: NumPy image (RGB) containing doctor and scanner.
            doctor_kp: Doctor keypoints returned by the model.
            doctor_scores: Scores for doctor keypoints returned by the model.
            thr: Threshold to consider arm as successfully detected, default 0.5.
        
        Returns:
            scanner_coords: Coordinate of the scanner edge, None if no/bad detection.
        """
        # Identify coordinate of wrist holding the scanner (doctor).
        elbow_coords, wrist_coords = self.find_scanner_hand(doctor_kp, doctor_scores, thr)
        if elbow_coords is None:
            return None
        
        # Load GT instead of cropping/running detection.
        frame_key = str(frame_id)
        frame_offset = np.asarray(self.offsets[frame_key])
        gt_masks = self.masks[frame_key]
                
        scanner_body_mask = [(gt_masks[0].astype(np.uint8)) * 255]
        scanner_tail_mask = [(gt_masks[1].astype(np.uint8)) * 255]
        scanner_body_cnt = []
        scanner_tail_cnt = []
        for mask in scanner_body_mask:
            cnt, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # cnt = [shrunk * 4 for shrunk in cnt]
            scanner_body_cnt += cnt # find_contours(mask)
        for mask in scanner_tail_mask:
            cnt, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # cnt = [shrunk * 4 for shrunk in cnt]
            scanner_tail_cnt += cnt # find_contours(mask)
        def get_area(contour):
            return cv2.contourArea(contour)
        scanner_body_cnt.sort(key=get_area)
        scanner_tail_cnt.sort(key=get_area)
        
        debug_frame = cv2.hconcat([scanner_body_mask[0], scanner_tail_mask[0]])
        debug_q.put(debug_frame)
        
        if len(scanner_body_cnt) == 0:
            # debug_q.put(scanner_frame)
            return None
        # cv2.drawContours(scanner_frame, [scanner_body_cnt[0]], -1, (255,0,0), 3, cv2.LINE_AA)
        if len(scanner_tail_cnt) == 0:
            # cv2.drawContours(debug_frame, [scanner_body_cnt[0]], -1, 255, thickness=-1)
            # debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)
            idx, _ = self.find_shape_endpoints(scanner_body_cnt[0], np.array([[wrist_coords - frame_offset]]))
            # cv2.circle(scanner_frame, np.array(wrist_coords - frame_offset, dtype=int), 5, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.circle(scanner_frame, scanner_body_cnt[0][idx][0], 5, (255,255,0), 2, cv2.LINE_AA)
            # debug_q.put(scanner_frame)
            return scanner_body_cnt[0][idx][0] + frame_offset
        
        idx, _ = self.find_shape_endpoints(scanner_body_cnt[0], scanner_tail_cnt[0])
        
        # cv2.circle(scanner_frame, scanner_body_cnt[0][idx][0], 5, (255,255,0), 2, cv2.LINE_AA)
        # cv2.drawContours(scanner_frame, [scanner_tail_cnt[0]], -1, (0,255,0), 1, cv2.LINE_AA)
        # debug_q.put(scanner_frame)
        
        return scanner_body_cnt[0][idx][0] + frame_offset
    
    # Find existing arms to identify the scanner hand (higher hand).
    def find_scanner_hand(self, kp, score, thr=0.5):
        elbow_coords, wrist_coords = None, None
        if score[7] > thr and score[9] > thr:  # check left arm
            elbow_coords, wrist_coords = kp[7], kp[9]
        if score[8] > thr and score[10] > thr:  # check right arm
            if wrist_coords is None or wrist_coords[1] > kp[10][1]:  # only assign if a higher hand
                elbow_coords , wrist_coords = kp[8], kp[10]
        return elbow_coords, wrist_coords
    
    def find_shape_endpoints(self, cnt1, cnt2):
        pts1 = cnt1.reshape(-1, 2)
        pts2 = cnt2.reshape(-1, 2)
        
        # Find closest point pairs between both contours (joint point).
        diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        idx1_joint, idx2_joint = np.unravel_index(np.argmax(dist_sq), dist_sq.shape)
        return idx1_joint, idx2_joint
    
        # joint_pt1 = pts1[idx1_joint]
        # joint_pt2 = pts2[idx2_joint]

        # # Find the furthest points from the joint point.
        # dists_to_joint1 = np.linalg.norm(pts1 - joint_pt1, axis=1)
        # idx1 = np.argmax(dists_to_joint1)

        # dists_to_joint2 = np.linalg.norm(pts2 - joint_pt2, axis=1)
        # idx2 = np.argmax(dists_to_joint2)
        # return idx1, idx2