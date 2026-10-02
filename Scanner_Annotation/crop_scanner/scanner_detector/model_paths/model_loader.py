from os import path
from rtmlib import RTMDet, RTMPose
from .MyRTMDet import MyRTMDet
from .rfdetr_onnx import RFDETR_ONNX


model_path_dir = path.dirname(path.abspath(__file__))

def load_det_model(filename, input_size, backend, device):
    det_model_path = path.join(model_path_dir, filename)
    det_model = MyRTMDet(onnx_model=det_model_path,
                       model_input_size=input_size,
                       backend=backend,
                       device=device)
    return det_model

def load_pose_model(filename, input_size, backend, device):
    pose_model_path = path.join(model_path_dir, filename)
    pose_model = RTMPose(onnx_model=pose_model_path,
                        model_input_size=input_size,
                        backend=backend,
                        device=device)
    return pose_model

def load_scanner_seg_model(filename, device='cuda'):
    seg_model_path = path.join(model_path_dir, filename)
    seg_model = RFDETR_ONNX(onnx_model_path=seg_model_path, device=device)
    return seg_model