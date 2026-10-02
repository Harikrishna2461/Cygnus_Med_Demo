"""
Stage 4: Locate scanner on the leg

Takes in RGB frame, use the patient's pose and scanner position to approximate
potential locations the scanner could be scanning. Previous positions and
tie-break algorithm are used to determine the segment and position on the leg.

The results are updated to the PositionMemory.
"""
import threading
import traceback
import queue
import cv2
import numpy as np

from .position_memory import PositionMemory
from .utils import find_skin_hsv, hsv_step_function

TIEBREAK_THR = 10000  # distance threshold to perform tie break
SEGMENTS_NAME = ['left thigh', 'left calf', 'right thigh', 'right calf']

class SegmentDetector:
    def __init__(self, position_memory: PositionMemory, barrier, exit_event):
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
    
    def start(self):
        self.worker_thread.start()
        print("    SegmentDetector thread started.")
    
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
                # Locate scanner relative to leg using found results.
                keypoints, scores = self.position_memory.get_keypoints()
                p_kp, p_sc= keypoints[0], scores[0]
                scanner_pos = self.position_memory.get_scanner_pos()
                segment_results = self.find_segment(p_kp, p_sc, scanner_pos, rgb_frame)
                self.position_memory.update_segment(segment_results)
                
                self.barrier.wait()
            
            except threading.BrokenBarrierError:
                break
            except Exception:
                print(f"ERROR in SegmentDetector thread:")
                traceback.print_exc()
                self.exit_event.set()
                self.barrier.abort()
                break
        print("    SegmentDetector thread closed.")
    
    def find_segment(self, patient_keypoints, patient_scores, scanner_pos, frame):    
        # Break if any of the leg keypoints are missing (score == -1).
        if np.sum(patient_scores[11:17] < 0):
            return None
        # Break if scanner is not found ((-1, -1)).
        if scanner_pos[0] == -1:
            return None
        
        # Organize points to segments: left thigh, left calf, right thigh, right calf.
        segments = np.asarray([[patient_keypoints[11], patient_keypoints[13]],
                            [patient_keypoints[13], patient_keypoints[15]],
                            [patient_keypoints[12], patient_keypoints[14]],
                            [patient_keypoints[14], patient_keypoints[16]]])
        # Given A, B and P, projection scalar t = (AP * AB) / (AB * AB).
        seg_diffs = segments[:, 1] - segments[:, 0]
        pt_diffs = scanner_pos - segments[:, 0]
        t = np.sum(pt_diffs * seg_diffs, axis=1) / np.sum(seg_diffs * seg_diffs, axis=1)
        proj = segments[:, 0] + t[:, None] * seg_diffs
        proj[:, 0] = np.clip(proj[:, 0], 0, 1279)
        proj[:, 1] = np.clip(proj[:, 1], 0, 719)
        proj_dist = np.linalg.norm(scanner_pos - proj, axis=1)
        
        on_segment = (t >= 0) & (t <= 1)  # mask where projection falls within the line
        masked_dist = np.where(on_segment, proj_dist, np.inf)  # set to inf if not on segment
        lowest_two_idx = np.argpartition(masked_dist, 2)[0:2]
        best_segment_idx = lowest_two_idx[0]
        test_idx = lowest_two_idx[1]
        
        segment_pos = proj[best_segment_idx].astype(int)
        test_pos = proj[test_idx].astype(int)
        
        # Perform tie break if best projections have similar lengths and are on opposite legs.
        tiebreak_flag = False
        if (proj_dist[lowest_two_idx[1]] - proj_dist[lowest_two_idx[0]] < TIEBREAK_THR
            and (best_segment_idx // 2) != (test_idx // 2)):
            tiebreak_flag = True
            
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            skin_hsv = find_skin_hsv(hsv_frame, patient_keypoints, patient_scores)  # find leg pixels HSV value.
            
            main_pt = proj[best_segment_idx].astype(int)
            main_hsv = hsv_frame[main_pt[1], main_pt[0], :].astype(int)
            if skin_hsv is not None:
                if abs(main_hsv[0] - skin_hsv[0]) < 20 and abs(main_hsv[1] - skin_hsv[1]) < 60:  # only if initial is on skin
                    main_pt = hsv_step_function(hsv_frame, main_pt, scanner_pos)
                main_scanner_pt = hsv_step_function(hsv_frame, scanner_pos, main_pt)
                sub_pt = proj[test_idx].astype(int)
                sub_hsv = hsv_frame[sub_pt[1], sub_pt[0], :].astype(int)
                if abs(sub_hsv[0] - skin_hsv[0]) < 20 and abs(sub_hsv[1] - skin_hsv[1]) < 60:  # only if initial is on skin
                    sub_pt = hsv_step_function(hsv_frame, sub_pt, scanner_pos)
                sub_scanner_pt = hsv_step_function(hsv_frame, scanner_pos, sub_pt)
            
            main_dist = np.linalg.norm(main_scanner_pt - main_pt)
            sub_dist = np.linalg.norm(sub_scanner_pt - sub_pt)
            # Swap based on distance after stepping along skin to scanner.
            if sub_dist < main_dist:
                temp_idx = best_segment_idx
                best_segment_idx = test_idx
                test_idx = temp_idx
                segment_pos = sub_pt
                test_pos = main_pt
            else:
                segment_pos = main_pt
                test_pos = sub_pt
            if abs(main_dist - sub_dist) > TIEBREAK_THR:
                tiebreak_flag = False
        # print(best_segment_idx)
        return (best_segment_idx, segment_pos, t[best_segment_idx], test_idx, test_pos, t[test_idx], tiebreak_flag)