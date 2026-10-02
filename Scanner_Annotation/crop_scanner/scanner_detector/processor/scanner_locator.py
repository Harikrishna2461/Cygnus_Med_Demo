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

import cv2
import numpy as np
import time

from .position_memory import PositionMemory
from ..model_paths.model_loader import load_scanner_seg_model
from .utils import find_contours, contour_touches_contour
from ..shared import debug_q


# Constants for cropping window around scanner.
CROP_W, CROP_H = 168, 168  # match RF-DETR
CROP_OFFSET = 80

class ScannerLocator:
    def __init__(self, position_memory: PositionMemory, barrier, exit_event, device='cuda'):
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
        if device == 'cuda':
            self.scanner_seg_model = load_scanner_seg_model(filename='scanner_segmentation_model.onnx', device=device)
        else:
            self.scanner_seg_model = HSVScannerSegmenter()
        self.worker_thread = threading.Thread(target=self.run, daemon=True)
    
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
                    rgb_frame = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue
                # Run scanner detection with keypoints/scores.
                keypoints, scores = self.position_memory.get_keypoints()
                d_kp, d_sc = keypoints[1], scores[1]
                scanner_pos = self.find_scanner_pos(rgb_frame, d_kp, d_sc)
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
    
    def find_scanner_pos(self, rgb_frame, doctor_kp, doctor_scores, thr=0.5):
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
        
        scanner_frame, crop_center, frame_offset = self.crop_scanner_frame(rgb_frame, elbow_coords, wrist_coords)
        if frame_offset is None:
            return None
        
        debug_q.put(scanner_frame)
    
    # Find existing arms to identify the scanner hand (higher hand).
    def find_scanner_hand(self, kp, score, thr=0.5):
        elbow_coords, wrist_coords = None, None
        if score[7] > thr and score[9] > thr:  # check left arm
            elbow_coords, wrist_coords = kp[7], kp[9]
        if score[8] > thr and score[10] > thr:  # check right arm
            if wrist_coords is None or wrist_coords[1] > kp[10][1]:  # only assign if a higher hand
                elbow_coords , wrist_coords = kp[8], kp[10]
        return elbow_coords, wrist_coords
    
    # Crop the window (224x224) containing scanner as well as coordinate of top left corner.
    def crop_scanner_frame(self, rgb_frame, elbow_coords, wrist_coords):       
        # Crop the scanner window.
        h, w, _ = rgb_frame.shape
        v = wrist_coords - elbow_coords
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-6:  # wrist/elbow keypoints coincide (degenerate pose) - skip this frame
            return None, None, None
        v /= v_norm
        crop_center = wrist_coords + v * CROP_OFFSET
        if crop_center[0] > w-1 or crop_center[1] > h-1:  # scanner out of bounds
            return None, None, None
        x_1 = int(crop_center[0] - CROP_W/2)
        x_2 = x_1 + CROP_W
        y_1 = int(crop_center[1] - CROP_H/2)
        y_2 = y_1 + CROP_H
        if x_1 < 0:
            x_1, x_2 = 0, CROP_W
        if x_2 > w:
            x_1, x_2 = w - CROP_W, w
        if y_1 < 0:
            y_1, y_2 = 0, CROP_H
        if y_2 > h:
            y_1, y_2 = h - CROP_H, h
        # x_1, x_2, y_1, y_2 = clip_to_frame(x_1, x_2, y_1, y_2, h, w)
        if x_1 == x_2 or y_1 == y_2:
            return None, None, None
        
        cropped_window = rgb_frame[y_1:y_2, x_1:x_2]
        offset = np.array([x_1, y_1], dtype=np.float32)  # offset to match original
        return cropped_window, crop_center, offset
    
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

class HSVScannerSegmenter:
    """
    Running RF-DETR model without GPU is very slow.
    This classes utilizes OpenCV functions for an algorithm to detect
    white scanner body using HSV channel and threshold masking.
    """
    def predict(self, rgb_scanner_frame, thr):
        hsv_frame = cv2.cvtColor(rgb_scanner_frame, cv2.COLOR_RGB2HSV)
        hsv_frame[:,:,0] = hsv_frame[:,:,0] // 10 * 10
        hsv_frame[:,:,1] = hsv_frame[:,:,1] // 40 * 40
        hsv_frame[:,:,2] = hsv_frame[:,:,2] // 40 * 40
        lower = np.array([0, 0, 100])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Shrinking mask to drown thin, long cables.
        kernel_size = 3  # adjust as needed (e.g., (3, 3), (5, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        shrunk_mask = cv2.erode(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(shrunk_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area_min_thr = 30
        area_max_thr = 4000
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_min_thr]
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) < area_max_thr]
        
        # Default fallback return values.
        labels = np.array([1], dtype=int)
        masks = np.zeros((int(CROP_H / 4), int(CROP_W / 4)), dtype=np.uint8)
        if not contours:
            return None, labels, None, np.array([masks])
        
        # Find the contour closest to the center of the cropped frame.
        crop_center = [int(CROP_H / 2), int(CROP_W / 2)]
        def get_center_offset(contour):
            val = cv2.pointPolygonTest(contour, crop_center, True)
            return abs(val)
        contours.sort(key=get_center_offset)
        
        hsv_contour = contours[0]
        hsv_mask = np.zeros_like(mask)
        cv2.drawContours(hsv_mask, [hsv_contour], -1, 255, thickness=-1)

        grayscale_image = cv2.cvtColor(rgb_scanner_frame, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(grayscale_image)
        _, thresholded_image = cv2.threshold(enhanced_image, 190, 255, cv2.THRESH_BINARY)
        thr_cnt, _ = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        scanner_contour = None
        for cnt in thr_cnt:
            if contour_touches_contour(cnt, hsv_mask):
                scanner_contour = cnt
                break
        if scanner_contour is None:
            return None, labels, None, np.array([masks])
        
        scanner_contour = np.asarray([shrunk / 4 for shrunk in scanner_contour], dtype=int)
        # Draw the best contour and return as mask.
        cv2.drawContours(masks, [scanner_contour], -1, 255, thickness=-1)
        return None, labels, None, np.array([masks])