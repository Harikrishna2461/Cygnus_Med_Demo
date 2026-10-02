"""
Helper classes and functions for the pose estimation and scanner localization algorithm.
"""
import cv2
import numpy as np


def calculate_iou(bbox1, bbox2):
    # Find intersection (A - top/left, B - bottom/right)
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[2], bbox2[2])
    yB = min(bbox1[3], bbox2[3])

    # Calculate area of intersection.
    intersection = max(0, xB - xA) * max(0, yB - yA)

    # Calculate are of union.
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = (bbox1_area + bbox2_area - intersection)
    
    # Calculate IoU.
    iou = intersection / union
    return iou

def clip_to_frame(x_1, x_2, y_1, y_2, h, w):
    x_1 = 0 if x_1 < 0 else x_1
    x_2 = w-1 if x_2 > w-1 else x_2
    y_1 = 0 if y_1 < 0 else y_1
    y_2 = h-1 if y_2 > h-1 else y_2
    return x_1, x_2, y_1, y_2

def find_contours(mask):
    # Constants for filtering contours.
    CNT_MIN_AREA = 10
    CNT_MAX_AREA = 1000
    
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > CNT_MIN_AREA]
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < CNT_MAX_AREA]
    return contours

def contour_touches_contour(cntA, mask):
    mask_h, mask_w = mask.shape
    def get_neighbour_pixels(x, y):
        x_min = 0 if x-1 < 0 else x-1
        x_max = mask_w-1 if x+1 > mask_w-1 else x+1
        y_min = 0 if y-1 < 0 else y-1
        y_max = mask_h-1 if y+1 > mask_h-1 else y+1
        return mask[y_min:y_max, x_min:x_max]
    for p in cntA:
        x, y = p[0]
        neighbours = get_neighbour_pixels(x,y)
        if np.sum(neighbours):
            return True
    return False

def closest_furthest_points(cnt1, cnt2):
    A = cnt1.reshape(-1, 2)
    B = cnt2.reshape(-1, 2)

    diff = A[:, None, :] - B[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

    min_idx = np.unravel_index(np.argmax(dists), dists.shape)
    max_idx = np.unravel_index(np.argmax(dists), dists.shape)
    
    return min_idx[0], min_idx[1], max_idx[0], max_idx[1]

def pixels_on_line(pt1, pt2, ratio=1.0):
    x_change = pt2[0] - pt1[0]
    y_change = pt2[1] - pt1[1]
    if x_change == 0 and y_change == 0:
        return np.asarray([pt1]).astype(int)
    
    transpose = False
    if abs(y_change) > abs(x_change):
        transpose = True
        slope = (pt2[0] - pt1[0]) / (pt2[1] - pt1[1])
        end_step = int(pt1[1] + ratio * (pt2[1] - pt1[1])) + 1
        base = np.arange(int(pt1[1]), end_step)
        intercept = pt1[0] - slope * pt1[1]
        result = np.full(len(base), intercept)
    else:
        slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        end_step = int(pt1[0] + ratio * (pt2[0] - pt1[0])) + 1
        increment = 1
        if pt1[0] > end_step:
            increment = -1
        base = np.arange(int(pt1[0]), end_step, increment)
        intercept = pt1[1] - slope * pt1[0]
        result = np.full(len(base), intercept)
    for i, step in enumerate(base):
        result[i] += step * slope
    result = np.round(result)
    if transpose:
        results = np.transpose(np.vstack([result, base])).astype(int)
    else:
        results = np.transpose(np.vstack([base, result])).astype(int)
    return results

def get_leg_pixels(kp_list):
    ratio = 1.0
    pixels = np.array([], dtype=int).reshape(0, 2)
    pixels = np.concatenate((pixels, pixels_on_line(kp_list[2], kp_list[0], ratio)))
    pixels = np.concatenate((pixels, pixels_on_line(kp_list[2], kp_list[4], ratio)))
    pixels = np.concatenate((pixels, pixels_on_line(kp_list[3], kp_list[0], ratio)))
    pixels = np.concatenate((pixels, pixels_on_line(kp_list[3], kp_list[5], ratio)))
    pixels[:, 0] = np.clip(pixels[:, 0], 0, 1279)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, 719)
    return pixels

def find_skin_hsv(hsv_frame, patient_keypoints, patient_scores):
    skin_hsv = None
    if not np.sum(patient_scores[11:17] < 0):
        leg_pixels = get_leg_pixels(patient_keypoints[11:17])
        hsv_pixels = {}
        for pixel in leg_pixels:
            hsv_rounded = hsv_frame[pixel[1], pixel[0], :] // 5 * 5
            try:
                hsv_pixels[tuple(hsv_rounded)] += 1
            except KeyError:
                hsv_pixels[tuple(hsv_rounded)] = 1
        skin_hsv = np.asarray(max(hsv_pixels, key=hsv_pixels.get), dtype=int)  # mode hsv value
    return skin_hsv

def hsv_step_function(hsv_frame, start_pt, end_pt):
    pt = start_pt.astype(int)
    line_pixels = pixels_on_line(pt, end_pt)
    h, w, _ = hsv_frame.shape  # for clipping coordinates
    line_pixels[:, 0] = np.clip(line_pixels[:, 0], 0, w - 1)
    line_pixels[:, 1] = np.clip(line_pixels[:, 1], 0, h - 1)
    hsv = hsv_frame[start_pt[1], start_pt[0], :].astype(int)  # HSV is uint8 which overflows
    for point in line_pixels:
        test_hsv = hsv_frame[point[1], point[0], :].astype(int)
        # Compare hue and saturation and stop stepping if change is large.
        if abs(hsv[0] - test_hsv[0]) > 20 or abs(hsv[1] - test_hsv[1]) > 40:
            break
        hsv = test_hsv
        pt = point
    return pt