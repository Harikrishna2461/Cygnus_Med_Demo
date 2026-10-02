from os import path
from .rfdetr_onnx import RFDETR_ONNX


model_path_dir = path.dirname(path.abspath(__file__))

def load_scanner_seg_model(filename, device='cuda'):
    seg_model_path = path.join(model_path_dir, filename)
    seg_model = RFDETR_ONNX(onnx_model_path=seg_model_path, device=device)
    return seg_model