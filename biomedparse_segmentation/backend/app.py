"""
BiomedParse Vein Segmentation — video app.

Flow:
  1. POST /api/upload           -> upload a video, kicks off background processing
  2. GET  /api/status/<job_id>  -> poll progress (fps/frame_count detected + % done)
  3. GET  /api/video/<job_id>   -> stream the annotated (green vein outline) H.264 mp4
  4. GET  /api/download_coco/<job_id> -> download COCO 1.0 zip (images/ + annotations/instances_default.json)
     ready to import into CVAT as an "Images" task, format "COCO 1.0".

Vein-only segmentation (no fascia) — see backend/engine.py.
FPS/frame-count are read from the actual uploaded video (backend/video_io.py:probe_video),
never hardcoded, so annotation timing/frame numbering stays correct for any input video.
"""
import os
import sys
import threading
import traceback
import uuid

from flask import Flask, request, jsonify, send_file, render_template

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # backend/ itself

from flask_cors import CORS

import numpy as np

import engine
import roi as roi_detect
import video_io
from coco_export import CocoVideoAnnotationBuilder

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend'),
    static_folder=os.path.join(BASE_DIR, 'frontend', 'static'),
)
CORS(app)

UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

VEIN_THRESHOLD = 0.5

# job_id -> status dict
_jobs = {}
_jobs_lock = threading.Lock()

_model_lock = threading.Lock()


def _set_job(job_id, **kwargs):
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _process_video(job_id: str, input_path: str, video_name: str):
    try:
        _set_job(job_id, stage='probing', message='Reading video metadata...')
        meta = video_io.probe_video(input_path)
        fps = meta['fps']
        frame_count = meta['frame_count']
        width, height = meta['width'], meta['height']

        _set_job(
            job_id, stage='loading_model', message='Loading BiomedParse vein model...',
            fps=fps, frame_count=frame_count, meta_frame_count=meta['meta_frame_count'],
            width=width, height=height, duration_sec=meta['duration_sec'], processed=0,
        )

        with _model_lock:
            model = engine.load_model()

        _set_job(job_id, stage='detecting_roi', message='Detecting scan-area ROI (excluding machine UI chrome)...')
        x1, y1, x2, y2 = roi_detect.detect_roi(input_path, width, height)
        _set_job(job_id, roi=[x1, y1, x2, y2])

        out_video_path = os.path.join(OUTPUT_DIR, f'{job_id}_annotated.mp4')
        out_coco_path = os.path.join(OUTPUT_DIR, f'{job_id}_coco.zip')

        writer = video_io.H264VideoWriter(out_video_path, fps=fps, width=width, height=height)
        coco = CocoVideoAnnotationBuilder(video_name=video_name, fps=fps, width=width, height=height)

        _set_job(job_id, stage='processing', message='Segmenting veins frame by frame...')

        total_vein_frames = 0
        for idx, frame_rgb in video_io.iter_frames(input_path):
            # Segment only the true scan-area crop — running the model on the
            # full device screen (icons, black letterboxing) produces false
            # positives there, since it looks nothing like the ROI-cropped
            # frames the model was trained on. The resulting mask is in crop
            # coordinates; paste it back into a full-frame-sized canvas so
            # drawing/export stay in the original video's coordinate space.
            crop = frame_rgb[y1:y2, x1:x2]
            crop_mask = engine.segment_frame(model, crop, threshold=VEIN_THRESHOLD)
            mask = np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = crop_mask

            if mask.max() > 0:
                total_vein_frames += 1
            annotated = engine.draw_vein_contours(frame_rgb, mask)
            writer.write(annotated)
            coco.add_frame(idx, frame_rgb, mask)

            if idx % 5 == 0 or idx == frame_count - 1:
                _set_job(job_id, processed=idx + 1, vein_frames=total_vein_frames)

        writer.close()
        coco.write_zip(out_coco_path)

        _set_job(
            job_id, stage='done', message='Done.',
            processed=frame_count, vein_frames=total_vein_frames,
            video_url=f'/api/video/{job_id}', coco_url=f'/api/download_coco/{job_id}',
        )
    except Exception as e:
        traceback.print_exc()
        _set_job(job_id, stage='error', message=str(e))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    job_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    input_path = os.path.join(UPLOAD_DIR, f'{job_id}{ext}')
    file.save(input_path)

    with _jobs_lock:
        _jobs[job_id] = {
            'job_id': job_id, 'stage': 'queued', 'message': 'Queued...',
            'video_name': file.filename, 'processed': 0, 'frame_count': None,
        }

    thread = threading.Thread(target=_process_video, args=(job_id, input_path, file.filename), daemon=True)
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/api/status/<job_id>')
def status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({'error': 'Unknown job_id'}), 404
    return jsonify(job)


@app.route('/api/video/<job_id>')
def get_video(job_id):
    path = os.path.join(OUTPUT_DIR, f'{job_id}_annotated.mp4')
    if not os.path.exists(path):
        return jsonify({'error': 'Video not ready'}), 404
    return send_file(path, mimetype='video/mp4', conditional=True)


@app.route('/api/download_coco/<job_id>')
def download_coco(job_id):
    path = os.path.join(OUTPUT_DIR, f'{job_id}_coco.zip')
    if not os.path.exists(path):
        return jsonify({'error': 'Annotations not ready'}), 404
    with _jobs_lock:
        video_name = _jobs.get(job_id, {}).get('video_name', job_id)
    base = os.path.splitext(video_name)[0]
    return send_file(path, mimetype='application/zip', as_attachment=True,
                      download_name=f'{base}_vein_coco.zip')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
