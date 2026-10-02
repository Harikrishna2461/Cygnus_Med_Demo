# Missing Model Files

The following public models are excluded from this repository due to size.

## RTMDet-m (rtmdet-m-640.onnx)
Download from MMPose:
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet-m_640-8xb32_coco-person-9d5d88c0_20230824.zip
Extract and place: model_paths/rtmdet-m-640.onnx

## RTMPose-l (rtmpose-l-192.onnx)  
Download from MMPose:
https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-aic-coco_210e-256x192-f016ffe0_20230126.zip
Extract and place: model_paths/rtmpose-l-192.onnx

## SAM2.1 checkpoints (sam2.1_hiera_*.pt) + EdgeTAM (edgetam.pt)
These are auto-downloaded on first run by sam_segmentation.py — no manual action needed.

## scanner_segmentation_model*.onnx (custom trained)
Master copy located at: Cygnus_Med_Demo/Scanner_Annotation/crop_scanner/model_paths/
Copy from there into this folder's model_paths/ before running.
