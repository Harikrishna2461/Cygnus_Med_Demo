"""
New real-time vein classification endpoint using advanced detector
"""

import logging
import base64
import cv2
import numpy as np
from typing import Dict, Optional
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def create_vein_classification_endpoint(app, detector):
    """
    Create Flask endpoint for real-time vein classification
    
    Args:
        app: Flask application
        detector: UltrasoundVeinDetector instance
    """
    
    @app.route('/api/vision/classify-veins-realtime', methods=['POST'])
    def classify_veins_realtime():
        """
        Real-time vein classification endpoint
        
        Expected form:
        - file: Image file (JPEG/PNG)
        - fascia_y: Optional Y-coordinate of fascia (default: center)
        - return_json: Whether to return just JSON (default: true)
        """
        try:
            # Get image from request
            if 'file' not in request.files:
                return {"status": "error", "error": "No file provided"}, 400
            
            file = request.files['file']
            
            # Read image
            image_data = file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"status": "error", "error": "Invalid image"}, 400
            
            # Get optional parameters
            fascia_y = request.form.get('fascia_y', type=int)
            return_json = request.form.get('return_json', 'true').lower() == 'true'
            
            logger.info(f"Processing image: {frame.shape}")
            
            # Detect and classify
            result = detector.detect_and_classify_frame(frame, fascia_y)
            
            if not result.get("detections"):
                return {
                    "status": "no_veins",
                    "message": "No veins detected",
                    "detections": []
                }, 200
            
            # Prepare response
            response_data = {
                "status": result["status"],
                "num_veins": result["num_veins"],
                "detections": [
                    {
                        "id": i,
                        "classification": d["classification"],
                        "confidence": float(d["confidence"]),
                        "center": d["center"],
                        "bbox": d["bbox"],
                        "vein_type": d["vein_type"],
                        "fascia_distance": d["fascia_distance"],
                        "verified_by_claude": d.get("verified_by_claude", False)
                    }
                    for i, d in enumerate(result["detections"])
                ]
            }
            
            if return_json:
                return response_data, 200
            else:
                # Encode annotated frame
                success, buffer = cv2.imencode('.png', result["annotated_frame"])
                if success:
                    frame_b64 = base64.b64encode(buffer).tobytes().decode()
                    response_data["annotated_image"] = f"data:image/png;base64,{frame_b64}"
                
                return response_data, 200
        
        except Exception as e:
            logger.error(f"Error in vein classification: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}, 500
    
    @app.route('/api/vision/classify-veins-stream', methods=['POST'])
    def classify_veins_stream():
        """
        Stream-based vein classification for video files
        
        Expected form:
        - file: Video file
        - output: Whether to save output video (optional)
        - fascia_y: Optional Y-coordinate of fascia
        """
        try:
            if 'file' not in request.files:
                return {"status": "error", "error": "No file provided"}, 400
            
            file = request.files['file']
            
            # Save video temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                file.save(tmp.name)
                video_path = tmp.name
            
            fascia_y = request.form.get('fascia_y', type=int)
            save_output = request.form.get('output', 'false').lower() == 'true'
            
            logger.info(f"Processing video: {video_path}")
            
            # Determine output path
            output_path = None
            if save_output:
                import tempfile
                output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                output_path = output_file.name
            
            # Process video
            detector.process_video_stream(video_path, output_path, fascia_y)
            
            response = {
                "status": "success",
                "message": "Video processed successfully",
                "output_path": output_path
            }
            
            return response, 200
        
        except Exception as e:
            logger.error(f"Error in stream classification: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "error": str(e)}, 500
    
    logger.info("✓ Real-time vein classification endpoints created")
