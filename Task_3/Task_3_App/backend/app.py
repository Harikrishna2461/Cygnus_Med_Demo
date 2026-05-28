import os
os.environ['TF_CUDNN_USE_AUTOTUNE']       = '0'
os.environ['TF_XLA_FLAGS']               = '--tf_xla_auto_jit=0'
os.environ['TF_DISABLE_MKL']             = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS']= '1'

import json, uuid, base64, re, time, tempfile
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
import pandas as pd
import io

tf.config.optimizer.set_jit(False)

app = Flask(__name__)
CORS(app)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
FASCIA_SIZE = 384
MEAN_NP     = np.array([0.485, 0.456, 0.406], np.float32)
STD_NP      = np.array([0.229, 0.224, 0.225], np.float32)
SEQ_LEN     = 5
NEAR_PX     = 20
DATA_ROOT   = Path('/home/krish/vein_detection_task_3_training')

LABEL_COLOR = {
    'N1':      (50,  80,  255),
    'N2':      (50,  255, 100),
    'N3':      (255, 150,  50),
    'unknown': (180, 180, 180),
}

VLM_PROMPT = (
    "Ultrasound image. CYAN line = fascia boundary. "
    "Numbered yellow boxes = detected veins.\n"
    "Classify each numbered vein:\n"
    "- N3: vein is ABOVE the fascia (superficial)\n"
    "- N2: vein is ON or VERY CLOSE to fascia (GSV)\n"
    "- N1: vein is BELOW the fascia (deep)\n"
    "Reply ONLY with JSON: {\"1\": \"N2\", \"2\": \"N3\"}"
)

# in-memory store: upload_id -> {tmp_path, records}
_store: dict = {}

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading models...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

fascia_model = tf.keras.models.load_model(
    str(DATA_ROOT / 'output/fascia/checkpoints/unet_resnet50_tf_best.keras'), compile=False)
vein_model = tf.keras.models.load_model(
    str(DATA_ROOT / 'output/vein/checkpoints/unet_resnet50_vein_best.keras'), compile=False)
lstm_model = tf.keras.models.load_model(
    str(DATA_ROOT / 'output/tracking/lstm_motion_predictor.keras'), compile=False)
print("Models ready.")

# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(bgr, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    return ((cv2.resize(rgb, (size, size)) - MEAN_NP) / STD_NP)[None]

def predict_fascia(bgr):
    mask = np.argmax(fascia_model(preprocess(bgr, FASCIA_SIZE), training=False).numpy()[0], -1).astype(np.uint8)
    return cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

def predict_vein(bgr):
    return np.argmax(vein_model(preprocess(bgr, IMG_SIZE), training=False).numpy()[0], -1).astype(np.uint8)

def get_fascia_boundary(mask):
    fb = (mask == 1).astype(np.uint8)
    if fb.sum() < 50:
        return None, None
    top_y = np.full(IMG_SIZE, np.nan, np.float32)
    bot_y = np.full(IMG_SIZE, np.nan, np.float32)
    for x in range(IMG_SIZE):
        ys = np.where(fb[:, x] > 0)[0]
        if len(ys):
            top_y[x], bot_y[x] = float(ys.min()), float(ys.max())
    valid = ~np.isnan(top_y)
    if valid.sum() < 2:
        return None, None
    xs    = np.arange(IMG_SIZE, dtype=np.float32)
    top_y = uniform_filter1d(np.interp(xs, xs[valid], top_y[valid]), 15)
    bot_y = uniform_filter1d(np.interp(xs, xs[valid], bot_y[valid]), 15)
    return top_y, bot_y

def extract_detections(mask, min_area=50):
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dets = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        dets.append({
            'bbox':     (stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP],
                         stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]),
            'centroid': (float(centroids[i][0]), float(centroids[i][1])),
            'area':     int(a),
        })
    return dets

def vlm_classify(bgr, dets, top_y, bot_y, groq_client, vlm_model_name):
    if not dets or groq_client is None:
        return {}
    vis = bgr.copy()
    h0, w0 = bgr.shape[:2]
    sx, sy  = w0 / IMG_SIZE, h0 / IMG_SIZE
    if top_y is not None:
        pts = np.array([[int(x*sx), int(top_y[x]*sy)] for x in range(IMG_SIZE)], np.int32)
        cv2.polylines(vis, [pts], False, (0, 220, 220), 2)
    for idx, det in enumerate(dets, 1):
        x, y, w, h = det['bbox']
        cv2.rectangle(vis, (int(x*sx),int(y*sy)), (int((x+w)*sx),int((y+h)*sy)), (0,255,255), 2)
        cv2.putText(vis, str(idx), (int(x*sx), int(y*sy)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64    = base64.b64encode(buf.tobytes()).decode()
    try:
        resp = groq_client.chat.completions.create(
            model=vlm_model_name,
            messages=[{'role':'user','content':[
                {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}},
                {'type':'text','text':VLM_PROMPT},
            ]}],
            max_tokens=150, temperature=0.0)
        raw = resp.choices[0].message.content.strip()
        m   = re.search(r'\{[^}]+\}', raw)
        if m:
            return {int(k): v for k, v in json.loads(m.group()).items() if v in ('N1','N2','N3')}
    except Exception as e:
        print(f'VLM error: {e}')
    return {}

# ── Tracker ───────────────────────────────────────────────────────────────────
def bbox_iou(b1, b2):
    x1,y1,w1,h1=b1; x2,y2,w2,h2=b2
    ix = max(0, min(x1+w1,x2+w2) - max(x1,x2))
    iy = max(0, min(y1+h1,y2+h2) - max(y1,y2))
    return (ix*iy) / (w1*h1 + w2*h2 - ix*iy + 1e-6)

def match_dets(prev, curr, thresh=0.2):
    if not prev or not curr:
        return [], list(range(len(curr)))
    cost = np.array([[1.-bbox_iou(p['bbox'],c['bbox']) for c in curr] for p in prev], np.float32)
    rows, cols = linear_sum_assignment(cost)
    matches, matched = [], set()
    for r,c in zip(rows,cols):
        if cost[r,c] < (1.-thresh):
            matches.append((r,c)); matched.add(c)
    return matches, [j for j in range(len(curr)) if j not in matched]

class Tracker:
    def __init__(self, max_lost=5):
        self.tracks  = {}
        self.next_id = 0
        self.max_lost = max_lost

    def _predict(self):
        tids, seqs = [], []
        for tid, t in self.tracks.items():
            if len(t['history']) >= SEQ_LEN:
                tids.append(tid); seqs.append(t['history'][-SEQ_LEN:])
        pred = {}
        if seqs:
            ps = lstm_model(np.array(seqs, np.float32), training=False).numpy()
            for tid, p in zip(tids, ps):
                cx = float(np.clip(p[0],0,1))*IMG_SIZE; cy = float(np.clip(p[1],0,1))*IMG_SIZE
                w  = float(np.clip(p[2],0.01,1))*IMG_SIZE; h  = float(np.clip(p[3],0.01,1))*IMG_SIZE
                pred[tid] = (int(cx-w/2), int(cy-h/2), int(w), int(h))
        return pred

    def update(self, dets, fi):
        pred   = self._predict()
        active = sorted(self.tracks.keys())
        prev   = []
        for tid in active:
            d = dict(self.tracks[tid]['last_det'])
            if tid in pred: d['bbox'] = pred[tid]
            prev.append(d)
        matches, unmatched = match_dets(prev, dets)
        matched_prev = {p for p,_ in matches}
        for p_idx, c_idx in matches:
            tid = active[p_idx]; det = dets[c_idx]
            cx, cy = det['centroid']; x,y,w,h = det['bbox']
            self.tracks[tid].update({'last_det':det,'lost':0,'age':self.tracks[tid]['age']+1})
            self.tracks[tid]['history'].append([cx/IMG_SIZE,cy/IMG_SIZE,w/IMG_SIZE,h/IMG_SIZE])
        for i, tid in enumerate(active):
            if i not in matched_prev: self.tracks[tid]['lost'] += 1
        self.tracks = {tid:t for tid,t in self.tracks.items() if t['lost']<=self.max_lost}
        for c_idx in unmatched:
            det=dets[c_idx]; cx,cy=det['centroid']; x,y,w,h=det['bbox']
            self.tracks[self.next_id]={'last_det':det,'lost':0,'age':1,
                'history':[[cx/IMG_SIZE,cy/IMG_SIZE,w/IMG_SIZE,h/IMG_SIZE]]}
            self.next_id += 1
        records = []
        for tid, t in self.tracks.items():
            if t['lost'] == 0:
                det=t['last_det']; x,y,w,h=det['bbox']; cx,cy=det['centroid']
                records.append({'track_id':tid,'cx':round(cx,1),'cy':round(cy,1),
                    'x':x,'y':y,'w':w,'h':h,'area':det['area'],'age':t['age']})
        return records

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_frame(bgr, records, top_y, bot_y):
    vis = bgr.copy()
    h0, w0 = bgr.shape[:2]
    sx, sy  = w0/IMG_SIZE, h0/IMG_SIZE
    if top_y is not None:
        ov      = vis.copy()
        pts_top = np.array([[int(x*sx),int(top_y[x]*sy)] for x in range(IMG_SIZE)], np.int32)
        pts_bot = np.array([[int(x*sx),int(bot_y[x]*sy)] for x in range(IMG_SIZE)], np.int32)
        cv2.fillPoly(ov, [np.vstack([pts_top, pts_bot[::-1]])], (0, 200, 200))
        cv2.addWeighted(ov, 0.25, vis, 0.75, 0, vis)
        cv2.polylines(vis, [pts_top], False, (0, 220, 220), 2)
    for r in records:
        col  = LABEL_COLOR.get(r.get('label','unknown'), (180,180,180))
        x, y = int(r['x']*sx), int(r['y']*sy)
        w, h = int(r['w']*sx), int(r['h']*sy)
        cv2.rectangle(vis, (x,y), (x+w,y+h), col, 2)
        cv2.putText(vis, r.get('label','?'), (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return vis

def frame_to_b64(bgr):
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/upload', methods=['POST'])
def upload():
    f   = request.files.get('video')
    if not f:
        return jsonify({'error': 'No file'}), 400
    uid = str(uuid.uuid4())
    ext = Path(f.filename).suffix or '.mp4'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    f.save(tmp.name)
    _store[uid] = {'tmp_path': tmp.name, 'records': []}
    return jsonify({'upload_id': uid})


@app.route('/api/process/<upload_id>', methods=['GET'])
def process(upload_id):
    if upload_id not in _store:
        return jsonify({'error': 'Unknown upload_id'}), 404

    target_fps   = float(request.args.get('fps', 5))
    min_area     = int(request.args.get('min_area', 50))
    use_vlm      = request.args.get('use_vlm', 'true').lower() == 'true'
    groq_key     = request.args.get('groq_key', '')
    vlm_model_nm = request.args.get('vlm_model', 'meta-llama/llama-4-scout-17b-16e-instruct')

    groq_client = None
    if use_vlm and groq_key:
        try:
            from groq import Groq
            groq_client = Groq(api_key=groq_key)
        except Exception as e:
            print(f'Groq init error: {e}')

    tmp_path = _store[upload_id]['tmp_path']

    def generate():
        cap     = cv2.VideoCapture(tmp_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip    = max(1, round(src_fps / target_fps))
        n_proc  = max(1, total // skip)

        tracker  = Tracker()
        all_recs = []
        counts   = {'N1': 0, 'N2': 0, 'N3': 0, 'unknown': 0}
        raw_idx  = 0
        proc_idx = 0

        # Send metadata first
        yield f"data: {json.dumps({'type':'meta','total':n_proc,'src_fps':src_fps,'skip':skip})}\n\n"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if raw_idx % skip == 0:
                fascia_mask  = predict_fascia(frame)
                vein_mask    = predict_vein(frame)
                top_y, bot_y = get_fascia_boundary(fascia_mask)
                dets         = extract_detections(vein_mask, min_area)
                track_recs   = tracker.update(dets, proc_idx)

                vlm_labels = {}
                if use_vlm and groq_client and dets:
                    vlm_labels = vlm_classify(frame, dets, top_y, bot_y, groq_client, vlm_model_nm)

                for i, r in enumerate(track_recs, 1):
                    r['label'] = vlm_labels.get(i, 'unknown')
                    r['frame_idx'] = proc_idx
                    counts[r['label']] = counts.get(r['label'], 0) + 1
                    all_recs.append(r)

                annotated = draw_frame(frame, track_recs, top_y, bot_y)
                b64 = frame_to_b64(annotated)

                payload = {
                    'type':      'frame',
                    'frame_idx': proc_idx,
                    'total':     n_proc,
                    'image':     b64,
                    'counts':    counts.copy(),
                    'dets':      len(track_recs),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                proc_idx += 1

            raw_idx += 1

        cap.release()
        _store[upload_id]['records'] = all_recs

        yield f"data: {json.dumps({'type':'done','counts':counts,'total_processed':proc_idx})}\n\n"

    return Response(generate(),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache',
                             'X-Accel-Buffering': 'no'})


@app.route('/api/report/<upload_id>', methods=['GET'])
def download_report(upload_id):
    if upload_id not in _store:
        return jsonify({'error': 'Not found'}), 404

    records = _store[upload_id].get('records', [])
    if not records:
        return jsonify({'error': 'No records yet'}), 400

    df  = pd.DataFrame(records)
    cols = ['frame_idx','track_id','label','cx','cy','x','y','w','h','area','age']
    df  = df[[c for c in cols if c in df.columns]]

    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True,
                     download_name=f'vein_report_{upload_id[:8]}.csv')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
