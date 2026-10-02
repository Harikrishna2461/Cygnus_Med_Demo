import cv2
from os import path

from .. import shared
from ..processor.position_memory import PositionMemory

from rtmlib import draw_skeleton, draw_bbox


# Name of segments in vascular veins scanning.
SEGMENTS_NAME = ['left thigh', 'left calf', 'right thigh', 'right calf']

# Helpful things to draw the leg diagram.
path_dir = path.dirname(path.abspath(__file__))
leg_diagram_img_path = path.join(path_dir, 'small_leg.png')
leg_diagram = cv2.imread(leg_diagram_img_path)
leg_h, leg_w, _ = leg_diagram.shape
right_x = int(leg_w * 0.25)
left_x = int(leg_w * 0.75)
back_right_x = int(leg_w * 0.35)
back_left_x = int(leg_w * 0.65)
knee_y = int(leg_h * 0.6)
calf_len = leg_h - knee_y
SEGMENT_LEN = [knee_y, calf_len, knee_y, calf_len]
SEGMENT_POS = [(left_x, 0), (left_x, knee_y), (right_x, 0), (right_x, knee_y)]

class Visualizer():
    def __init__(self, config, position_memory: PositionMemory):
        self.config = config
        self.position_memory = position_memory
    
    def get_drawn_frames(self, frame):
        
        
        frame_dict = {}
        
        # Default frame to draw (or not) on.
        img_show = frame
        
        # Small calculation to identify patient orientation using hip coordinates.
        patient_ori = "front"
        if self.position_memory.keypoints[0][11][0] - self.position_memory.keypoints[0][12][0] < 0:
            patient_ori = "back"
        
        # Draw bounding boxes.
        if self.config['draw_bbox'] and self.position_memory.has_found_bbox:
            img_show = draw_bbox(img_show, self.position_memory.bboxes, (0,255,255))
        
        # Draw person.
        if self.config['draw_skeleton']:
            # TODO: custom draw function, up for implementation
            img_show = draw_skeleton(img_show,
                                    self.position_memory.keypoints,
                                    self.position_memory.scores,
                                    kpt_thr=0.5)
        # Draw scanner related components.
        if self.position_memory.segment_id is not None:
            if self.config['draw_scanner_location']:
                cv2.circle(img_show, self.position_memory.scanner_pos, 10, (255, 0, 255), 2)  # scanner
            if self.config['draw_scanner_projection'] and self.position_memory.scanner_pos[0] != -1: # and self.position_memory.found_scanner_counter > 10:
                cv2.line(img_show, self.position_memory.scanner_pos, self.position_memory.segment_pos, (255, 0, 255), 2)
                # test pos
                cv2.line(img_show, self.position_memory.scanner_pos, self.position_memory.test_pos, (255, 255, 0), 2)
            if self.config['draw_result_msg']:
                result_msg = "Segment: " + patient_ori + " " + SEGMENTS_NAME[self.position_memory.segment_id]
                result_msg += ", Percentage: {:.2f}%".format(self.position_memory.segment_dist * 100)
                cv2.putText(img=img_show, text=result_msg, org=self.position_memory.scanner_pos,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        
        # If the algorithm failed, draw reason.
        if self.position_memory.error != "":
            cv2.putText(img=img_show, text=self.position_memory.error, org=(20, 40),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        
        frame_dict['webcam'] = img_show
        
        # Draw leg related components.
        if self.config['draw_leg_diagram']:
            leg_copy = leg_diagram.copy()
            if self.position_memory.segment_id is not None:
                best_segment_idx = self.position_memory.segment_id
                scanner_pos = SEGMENT_POS[best_segment_idx]
                if best_segment_idx == 1 and patient_ori == "back":
                    scanner_pos = (back_left_x, scanner_pos[1])
                if best_segment_idx == 3 and patient_ori == "back":
                    scanner_pos = (back_right_x, scanner_pos[1])
                scanner_pos = (scanner_pos[0], scanner_pos[1] + int(SEGMENT_LEN[best_segment_idx] * self.position_memory.segment_dist))
                cv2.circle(leg_copy, scanner_pos, 10, (255,0,255), 2)
            frame_dict['leg_diagram'] = leg_copy
            
        return frame_dict
            
        