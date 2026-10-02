import numpy as np
import threading


'''
PositionMemory class keeps track of important data such as keypoints and segments.
No calculation should be done other than comparing update thresholds.
'''
class PositionMemory():
    BBOX_LIFETIME = 30
    
    KP_LIFETIME = 40  # number of missing frames to kill the point
    
    SCANNER_LIFETIME = 30
    SCANNER_PATIENCE = 20
    
    SEGMENT_LIFETIME = 30
    BAD_SEGMENT_DIST = 100
    BAD_SEGMENT_LIFETIME = 10
    
    def __init__(self, device):
        self.device = device
        
        if self.device != 'cuda':
            self.SCANNER_PATIENCE = 0
            self.SCANNER_LIFETIME = 10
        
        # BBOX related (object detection)
        self.has_found_bbox = False
        self.bboxes = None
        self.pose_blacklist = []
        self.seg_blacklist = []
        self.bad_bbox_count = 0
        self.box_lock = threading.Lock()
        
        # KP/SCORE related (pose estimation)
        self.keypoints = np.zeros((2,17,2))
        self.scores = np.full((2,17), -1, dtype=np.float64)
        self.missing_kp_counter = np.zeros((2,17))
        self.kp_lock = threading.Lock()
        
        # Scanner position related (segmentation)
        self.scanner_pos = np.full(2, -1)
        self.temp_scanner_pos = np.full(2, -1)
        self.missing_scanner_counter = self.SCANNER_LIFETIME + 1  # so that first run resets it
        self.found_scanner_counter = 0
        self.scanner_lock = threading.Lock()
        
        # Leg diagram related (leg segment decision)
        self.bad_segment_count = 0
        self.missing_segment_counter = 0
        self.segment_id = None
        self.segment_pos = None
        self.segment_dist = 0
        self.test_pos = None
        self.segment_lock = threading.Lock()
        
        # Info if algorithm failed
        self.error = ""
    
    """
    Getter functions for other classes accessing values/results.
    """
    def get_bboxes(self):
        with self.box_lock:
            return self.has_found_bbox, self.bboxes
        
    def get_bbox_blacklist(self):
        with self.box_lock:
            return self.pose_blacklist, self.seg_blacklist
    
    def get_keypoints(self):
        with self.kp_lock:
            return self.keypoints, self.scores
    
    def get_scanner_pos(self):
        with self.scanner_lock:
            return self.scanner_pos
    
    def get_segment(self):
        with self.segment_lock:
            return self.segment_id, self.segment_pos, self.segment_dist, self.test_pos
    
    """
    Setter funcitons.
    
    RACE CONDITION CASES:
    Since each of the setters are called in unique threads, locks are only needed
    for attributes which other setters and other classes may reference.
    E.g. keypoints, scores, bboxes
    
    NO RACE CONDITION CASES:
    Attributes used for update logic but are limited to a single setter function
    does not need a lock to reduce overhead, but one may consider implementing it
    for safety, though it requires careful management to not create bottlenecks.
    These should never be accessed by other setters or other classes.
    E.g. bad_bbox_count, missing_kp_counter, found_scanner_counter
    
    Note that locks are not implemented to keep threads synchronous, but rather
    just for data integrity where no read happens as a write is happening.
    
    For any in-place modifications, ensure you create a copy first and then
    replace the reference so that the scope of locks stay minimum. Integers and
    booleans are immutable so they do not require a copy, but still requires lock.
    """
    def update_bboxes(self, bboxes):
        with self.box_lock:
            self.has_found_bbox = True
            self.bboxes = bboxes
            self.bad_bbox_count = 0
    
    def reset_bbox_blacklist(self):
        with self.box_lock:
            self.pose_blacklist = []
            self.seg_blacklist = []
    
    def reset_patient_kp(self):
        with self.kp_lock:
            my_sc = self.scores.copy()
            my_sc[0,:] = -1
            self.scores = my_sc
    def reset_all_kp(self):
        with self.kp_lock:
            my_sc = self.scores.copy()
            my_sc[:,:] = -1
            self.scores = my_sc
    
    def update_keypoints(self, keypoints, scores, swap_flag, score_thr=0.5, dist_thr=20, weight=0.5):
        # Creating a copy so that in-place modification can happen.
        my_kp, my_sc = self.keypoints.copy(), self.scores.copy()
        
        # Force global update if bbox needed a reset.
        if len(self.seg_blacklist) > 0 or len(self.pose_blacklist) > 0:
            my_sc[:,:] = -1
        
        # Reset mechanism
        my_sc[self.missing_kp_counter > self.KP_LIFETIME] = -1
        
        if self.bad_bbox_count > self.BBOX_LIFETIME:
            # with self.box_lock:
            self.has_found_bbox = False
            self.pose_blacklist.append(self.bboxes[0].copy())  # blackbox one of them
        
        # Bad detection (missing shoulders -> cannot identify instances)
        if keypoints is None or scores is None:
            self.missing_kp_counter += 1  # increment all parts
            self.bad_bbox_count += 1
            return
        
        # Swap bboxes to order as patient, doctor.
        if swap_flag:
            found, curr_bboxes = self.get_bboxes()
            new_bboxes = [curr_bboxes[1], curr_bboxes[0]]
            self.update_bboxes(new_bboxes.copy())
        
        # Using masks to filter only good detections.
        old_mask = my_sc > score_thr
        new_mask = scores > score_thr
        overlap_mask = old_mask & new_mask
        old_kp_masked = np.where(overlap_mask[:,:,None], my_kp, np.nan)
        new_kp_masked = np.where(overlap_mask[:,:,None], keypoints, np.nan)
        
        # Compute distance between good old points and good new points.
        dist = np.sqrt(np.sum((old_kp_masked - new_kp_masked)**2, axis=-1))
        
        # Update point coordinates separately for each keypoint.
        dist_mask = dist < dist_thr
        dist_mask = dist_mask | ((self.scores < 0) & new_mask)  # include fresh points
        my_kp[dist_mask] = keypoints[dist_mask]
        my_sc[dist_mask] = scores[dist_mask]
        
        # Handle detected/missing parts counting.
        self.missing_kp_counter[dist_mask] = 0
        self.missing_kp_counter[~dist_mask] += 1  # increment missing parts
        p_sc, d_sc = my_sc[0], my_sc[1]
        if (np.sum(p_sc[11:17] < score_thr)  # missing any patient legs
            or (np.sum(d_sc[[7,9]] < score_thr)
                and np.sum(d_sc[[8,10]]< score_thr))):  # missing both doctor arms
            self.bad_bbox_count += 1
        else:
            # with self.box_lock:
            self.bad_bbox_count = 0
            self.pose_blacklist = []
            
        # Finally replacing reference for access.
        with self.kp_lock:
            self.keypoints, self.scores = my_kp, my_sc
    
    def update_scanner(self, scanner_pos, dist_thr=15, weight=0.3):
        # Note that no lock is needed here to read self.scanner_pos because
        # writing it only happens in this setter which is never called twice simultaneously.
        # Hence, locks are only needed for writing.
        
        # Reset mechanism.
        if self.missing_scanner_counter > self.SCANNER_LIFETIME:
            with self.scanner_lock:
                self.scanner_pos = np.full(2, -1)
            # with self.segment_lock:
            self.segment_id = None
        
        # Break early if no scanner found.
        if scanner_pos is None:
            self.missing_scanner_counter += 1
            self.found_scanner_counter = 0
            return
                
        # If self scanner position is missing then accumulate at temp position.
        if self.scanner_pos[0] == -1:
            temp_dist = np.linalg.norm(self.temp_scanner_pos - scanner_pos)
            self.temp_scanner_pos = scanner_pos.astype(np.int32)
            if temp_dist < dist_thr:
                self.found_scanner_counter += 1
            else:
                self.found_scanner_counter = 0
            
            if self.found_scanner_counter > self.SCANNER_PATIENCE:  # if considered reliable
                new_scanner_pos = self.temp_scanner_pos.copy()
                with self.scanner_lock:
                    self.scanner_pos = new_scanner_pos
                self.missing_scanner_counter = 0
                self.found_scanner_counter = 0
            return
        
        # Else, compare the distance and update only if it passes the threshold.
        scanner_dist = np.linalg.norm(self.scanner_pos - scanner_pos)
        if scanner_dist < dist_thr:
            new_scanner_pos = self.scanner_pos + weight * (scanner_pos - self.scanner_pos)
            new_scanner_pos = new_scanner_pos.astype(np.int32)
            with self.scanner_lock:
                self.scanner_pos = new_scanner_pos
            self.missing_scanner_counter = 0
            self.found_scanner_counter += 1
        else:
            self.missing_scanner_counter += 1
            # Try accumulating at temp.
            temp_dist = np.linalg.norm(self.temp_scanner_pos - scanner_pos)
            self.temp_scanner_pos = scanner_pos.astype(np.int32)
            if temp_dist < dist_thr:
                self.found_scanner_counter += 1
            else:
                self.found_scanner_counter = 0
                
            if self.found_scanner_counter > self.SCANNER_PATIENCE:  # if considered reliable
                new_scanner_pos = self.temp_scanner_pos.copy()
                with self.scanner_lock:
                    self.scanner_pos = self.temp_scanner_pos
                self.missing_scanner_counter = 0
                self.found_scanner_counter = 0
    
    def update_segment(self, segment_results):
        # Break early if no segement found (happens if scanner pos or patient kps are missing).
        if segment_results is None:
            self.missing_segment_counter += 1
            return
        
        # Parse results.
        segment_id, segment_pos, segment_dist, test_id, test_pos, test_dist, flag = segment_results
        
        # Bad detection (wrong person detected as patient).
        if np.linalg.norm(self.scanner_pos - segment_pos) > self.BAD_SEGMENT_DIST:
            self.bad_segment_count += 1
        else:
            # with self.box_lock:
            self.bad_segment_count = 0
            self.seg_blacklist = []
        
        # Reset mechanism.
        if self.missing_segment_counter > self.SEGMENT_LIFETIME:
            # with self.segment_lock:
            self.segment_id = None
        if self.bad_segment_count > self.BAD_SEGMENT_LIFETIME:
            # with self.segment_lock:
            self.segment_id = None
            found, bboxes = self.get_bboxes()
            # with self.box_lock:
            self.has_found_bbox = False
            self.seg_blacklist.append(bboxes[0].copy())
            # return  # return to ignore any results on current frame
        
        # If self segment is missing, or tie was not broken then just update with best.
        if self.segment_id is None or not flag:
            with self.segment_lock:
                self.segment_id = segment_id
                self.segment_dist = segment_dist
                self.segment_pos = segment_pos.astype(np.int32)
                self.test_pos = test_pos.astype(np.int32)
            self.missing_segment_counter = 0
            return
        
        # If both best and second best segements are on opposite legs, just increment.
        is_seg_same_leg = (self.segment_id // 2) == (segment_id // 2)
        is_test_same_leg = (self.segment_id // 2) == (test_id //2)
        if not is_seg_same_leg and not is_test_same_leg:
            self.missing_segment_counter += 1
            return
        
        # If both are facing the same direction and better one is on wrong leg, increment.
        # self.scanner_pos will never be -1 here (since segment is detected) but just in case.
        if self.scanner_pos[0] == -1:
            self.missing_segment_counter += 1
            return
        is_facing_same = ((self.scanner_pos[0] - segment_pos[0]) * (self.scanner_pos[0] - test_pos[0])) > 0
        if is_facing_same and not is_seg_same_leg:
            self.missing_segment_counter += 1
            return
        
        # Else, use the better one that lies on the same leg.
        if is_test_same_leg:
            with self.segment_lock:
                self.segment_id = test_id
                self.segment_dist = test_dist
                self.segment_pos = segment_pos.astype(np.int32)
                self.test_pos = test_pos.astype(np.int32)
            self.missing_segment_counter = 0
        if is_seg_same_leg:  # updating later to override if also same leg
            with self.segment_lock:
                self.segment_id = segment_id
                self.segment_dist = segment_dist
                self.segment_pos = segment_pos.astype(np.int32)
                self.test_pos = test_pos.astype(np.int32)
            self.missing_segment_counter = 0
    
    def get_results(self):
        results = {}
        results['segment_id'] = self.segment_id
        results['segment_dist'] = self.segment_dist
        results['bboxes'] = self.bboxes
        
        if self.keypoints[0][11][0] - self.keypoints[0][12][0] < 0:
            results['is_front'] = False
        else:
            results['is_front'] = True
        
        # If algorithm failed, provide reasons.
        failed_reason = ""
        self.scores[0, 11:17]
        if (np.sum(self.scores[0][11:17] < 0.5)):  # missing any patient legs
            failed_reason = "Failed to detect patient legs"
        elif (np.sum(self.scores[1][[7,9]] < 0.5)
                and np.sum(self.scores[1][[8,10]]< 0.5)):
            failed_reason = "Failed to detect scanning arm"
            # if len(self.pose_blacklist) > 0:  # no doctor or doctor arm not found
            #     failed_reason = "Doctor/Patient not found/poor detection"
            # elif len(self.seg_blacklist) > 0:
            #     failed_reason = "Patient not found/poor detection"
        elif not self.has_found_bbox:
            failed_reason = "Targets not found in scene"
        elif self.scanner_pos[0] == -1:  # scanner is not detected
            failed_reason = "Scanner/Scanning hand not detected"
        elif self.segment_id is None:
            failed_reason = "Bad scanner/patient location"
        
        results['error'] = failed_reason
        self.error = failed_reason
        
        return results
