"""
Pure foundation-model pipeline test.
NO rules, NO brightness gradients, NO blob detection.
Models used:
  - Grounding DINO  → fascia box + vein boxes
  - SAM ViT-Base    → precise segmentation masks
  - Groq VLM        → N1 / N2 / N3 classification

Run: python test_foundation_pure.py [groq_key]
Outputs annotated frames to test_output_foundation/
"""
import sys, os, cv2, json, re, base64, time
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

IMG_SIZE = 256
HERE     = Path(__file__).parent
DATA_DIR = HERE / 'vein_detection_task_3_training' / 'Data' / '1-Videos'
OUT_DIR  = HERE / 'test_output_foundation'
OUT_DIR.mkdir(exist_ok=True)

GROQ_KEY = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('GROQ_KEY', '')

# ── Load GDINO ────────────────────────────────────────────────────────────────
print("Loading Grounding DINO...")
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image as PILImage

gdino_proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).eval()
print("GDINO ready.")

# ── Load SAM ──────────────────────────────────────────────────────────────────
print("Loading SAM ViT-Base (downloading if needed)...")
from transformers import SamModel, SamProcessor
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").eval()
sam_proc  = SamProcessor.from_pretrained("facebook/sam-vit-base")
print("SAM ready.")

# ── Load Groq ─────────────────────────────────────────────────────────────────
groq_client = None
if GROQ_KEY:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_KEY)
    print("Groq client ready.")
else:
    print("No Groq key — classification step will be skipped.")


# ── GDINO helper ──────────────────────────────────────────────────────────────
def gdino_detect(bgr, text_prompt, box_threshold=0.25):
    """
    Run GDINO on bgr with given text prompt.
    Returns list of (x1,y1,x2,y2) in ORIGINAL pixel coords, and scores.
    """
    h0, w0 = bgr.shape[:2]
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inputs = gdino_proc(images=pil, text=text_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    try:
        results = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[(h0, w0)]
        )[0]
        keep   = results['scores'].cpu() >= box_threshold
        boxes  = results['boxes'][keep].cpu().numpy()
        scores = results['scores'][keep].cpu().numpy()
        labels = [results['labels'][i] for i in range(len(results['labels'])) if keep[i]]
    except TypeError:
        results = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold, text_threshold=box_threshold * 0.8,
            target_sizes=[(h0, w0)]
        )[0]
        boxes  = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        labels = results['labels']
    return boxes, scores, labels


# ── SAM helper ────────────────────────────────────────────────────────────────
def sam_segment(bgr, box_xyxy):
    """
    Segment bgr using SAM with a single bounding box prompt (x1,y1,x2,y2).
    Returns mask as (H, W) uint8 in original resolution.
    """
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    inputs = sam_proc(pil, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_proc.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    # masks[0] shape: (num_input_boxes, num_candidates, H, W) = (1, 3, H, W)
    iou_scores = outputs.iou_scores[0, 0].cpu().numpy()  # shape (3,)
    best       = int(np.argmax(iou_scores))
    return masks[0][0, best].numpy().astype(np.uint8)    # (H, W)


# ── Fascia from GDINO + SAM ───────────────────────────────────────────────────
def detect_fascia(bgr):
    """
    1. GDINO: find the fascia band.
    2. SAM: segment it → per-column (top_y, bot_y) arrays in IMG_SIZE coords.
    Returns (top_y, bot_y) or (None, None).
    """
    h0, w0 = bgr.shape[:2]
    boxes, scores, labels = gdino_detect(
        bgr,
        "fascia . bright horizontal hyperechoic band . connective tissue layer",
        box_threshold=0.20,
    )
    print(f"  GDINO fascia: {len(boxes)} boxes  scores={scores.tolist()}")

    # Pick widest, flattest box in 10–80% depth range.
    # Full-image boxes are fine — SAM correctly finds the fascia band even when
    # prompted with the whole scan (it segments the most prominent horizontal layer).
    best_box, best_score = None, -1.0
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        cy_norm = ((y1 + y2) / 2) / h0
        if cy_norm < 0.10 or cy_norm > 0.80:
            continue
        bw, bh = x2 - x1, max(y2 - y1, 1)
        flatness = (bw / w0) * (bw / bh)
        print(f"    box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) cy_norm={cy_norm:.2f} "
              f"flatness={flatness:.2f} score={sc:.3f}")
        if flatness > best_score:
            best_score = flatness
            best_box   = (x1, y1, x2, y2)

    if best_box is None:
        print("  GDINO fascia: no suitable box found -> returning None")
        return None, None

    print(f"  Best fascia box: {best_box}")
    mask = sam_segment(bgr, best_box)                # (H0, W0)
    mask_r = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    top_out = np.full(IMG_SIZE, np.nan, np.float32)
    bot_out = np.full(IMG_SIZE, np.nan, np.float32)
    for x in range(IMG_SIZE):
        ys = np.where(mask_r[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        ctr = float(np.median(ys))
        top_out[x] = max(0,            ctr - 8)
        bot_out[x] = min(IMG_SIZE - 1, ctr + 8)

    valid = ~np.isnan(top_out)
    print(f"  SAM fascia mask: {valid.sum()} valid columns")
    if valid.sum() < 10:
        print("  SAM mask too sparse → returning None")
        return None, None

    xs      = np.where(valid)[0].astype(np.float32)
    x_min   = int(xs.min()); x_max = int(xs.max())
    scan_xs = np.arange(x_min, x_max + 1, dtype=np.float32)
    top_out[x_min:x_max+1] = uniform_filter1d(
        np.interp(scan_xs, xs, top_out[valid]), 20).astype(np.float32)
    bot_out[x_min:x_max+1] = uniform_filter1d(
        np.interp(scan_xs, xs, bot_out[valid]), 20).astype(np.float32)
    return top_out, bot_out


# ── Veins from GDINO + SAM ────────────────────────────────────────────────────
def detect_veins(bgr):
    """
    1. GDINO: find vein candidates.
    2. SAM: refine each box → tight bbox + centroid.
    Returns list of det dicts.
    """
    h0, w0 = bgr.shape[:2]
    boxes, scores, labels = gdino_detect(
        bgr,
        "vein . dark oval blood vessel . anechoic structure",
        box_threshold=0.20,
    )
    print(f"  GDINO veins: {len(boxes)} boxes  scores={scores.tolist()}")

    img_area = h0 * w0
    dets = []
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        bw, bh = x2 - x1, max(y2 - y1, 1)
        box_area = bw * bh
        # Skip boxes covering >40% of image — whole-scan false positives
        if box_area > img_area * 0.40:
            print(f"    SKIP full-image box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) area={box_area/img_area:.0%}")
            continue
        # Skip extremely wide flat bars (not veins)
        if bw > bh * 5:
            print(f"    SKIP flat box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) bw/bh={bw/bh:.1f}")
            continue
        print(f"    vein box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) score={sc:.3f}")

        # Refine with SAM
        mask = sam_segment(bgr, (x1, y1, x2, y2))
        mask_r = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        ys_m, xs_m = np.where(mask_r > 0)
        if len(xs_m) < 4:
            # SAM returned empty mask — use GDINO box as-is
            nx  = int(x1 * IMG_SIZE / w0)
            ny  = int(y1 * IMG_SIZE / h0)
            nbw = max(4, int(bw * IMG_SIZE / w0))
            nbh = max(4, int(bh * IMG_SIZE / h0))
        else:
            nx  = int(xs_m.min()); ny  = int(ys_m.min())
            nbw = int(xs_m.max()) - nx
            nbh = int(ys_m.max()) - ny

        if nbw < 2 or nbh < 2:
            continue

        dets.append({
            'bbox':     (nx, ny, nbw, nbh),
            'centroid': (float(nx + nbw / 2), float(ny + nbh / 2)),
            'area':     int(nbw * nbh),
            'score':    float(sc),
        })

    # Simple NMS
    dets = sorted(dets, key=lambda d: d['score'], reverse=True)
    kept = []
    for d in dets:
        x1,y1,bw,bh = d['bbox']
        def iou(k):
            kx,ky,kw,kh = k['bbox']
            ix = max(0, min(x1+bw,kx+kw)-max(x1,kx))
            iy = max(0, min(y1+bh,ky+kh)-max(y1,ky))
            return (ix*iy)/(bw*bh+kw*kh-ix*iy+1e-6)
        if all(iou(k) < 0.3 for k in kept):
            kept.append(d)
    return kept


# ── VLM classification ────────────────────────────────────────────────────────
VLM_PROMPT = (
    "B-mode ultrasound image. CYAN lines = fascia band. "
    "Numbered white boxes = detected veins.\n"
    "Classify each numbered vein:\n"
    "  N3 = above fascia (superficial)\n"
    "  N2 = at / very near fascia (GSV)\n"
    "  N1 = below fascia (deep)\n"
    "Reply ONLY with JSON: {\"1\": \"N2\", \"2\": \"N3\"}"
)

def vlm_classify(bgr, dets, top_y, bot_y):
    if not groq_client or not dets:
        return {}
    vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)).copy()
    h0, w0 = vis.shape[:2]
    # Draw fascia
    if top_y is not None:
        xs_v = [x for x in range(IMG_SIZE) if not np.isnan(top_y[x])]
        if xs_v:
            pts_t = np.array([[x, int(top_y[x])] for x in xs_v], np.int32)
            pts_b = np.array([[x, int(bot_y[x])] for x in xs_v], np.int32)
            cv2.polylines(vis, [pts_t], False, (0,255,255), 2)
            cv2.polylines(vis, [pts_b], False, (0,255,255), 2)
    # Draw vein boxes
    for i, d in enumerate(dets, 1):
        x, y, bw, bh = d['bbox']
        cv2.rectangle(vis, (x,y), (x+bw,y+bh), (255,255,255), 2)
        cv2.putText(vis, str(i), (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
    b64 = base64.b64encode(buf.tobytes()).decode()
    try:
        resp = groq_client.chat.completions.create(
            model='meta-llama/llama-4-scout-17b-16e-instruct',
            messages=[{'role':'user','content':[
                {'type':'image_url','image_url':{'url':f'data:image/jpeg;base64,{b64}'}},
                {'type':'text','text':VLM_PROMPT},
            ]}],
            max_tokens=150, temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{[^}]+\}', raw)
        if m:
            return {int(k): v for k,v in json.loads(m.group()).items() if v in ('N1','N2','N3')}
    except Exception as e:
        print(f"  VLM error: {e}")
    return {}


# ── Draw output ───────────────────────────────────────────────────────────────
LABEL_COLOR = {'N1':(50,50,220),'N2':(50,200,50),'N3':(0,140,255),'?':(150,150,150)}

def draw_result(bgr, top_y, bot_y, dets, labels, fname_tag):
    vis = bgr.copy()
    h0, w0 = bgr.shape[:2]
    sx, sy = w0/IMG_SIZE, h0/IMG_SIZE

    if top_y is not None:
        xs_v = [x for x in range(IMG_SIZE) if not np.isnan(top_y[x])]
        if xs_v:
            pts_t = np.array([[int(x*sx), int(top_y[x]*sy)] for x in xs_v], np.int32)
            pts_b = np.array([[int(x*sx), int(bot_y[x]*sy)] for x in xs_v], np.int32)
            ov = vis.copy()
            cv2.fillPoly(ov, [np.vstack([pts_t, pts_b[::-1]])], (0,180,180))
            cv2.addWeighted(ov, 0.18, vis, 0.82, 0, vis)
            cv2.polylines(vis, [pts_t], False, (0,255,255), 2)
            cv2.polylines(vis, [pts_b], False, (0,255,255), 2)

    for i, d in enumerate(dets, 1):
        lbl = labels.get(i, '?')
        col = LABEL_COLOR.get(lbl, (150,150,150))
        x,y,bw,bh = d['bbox']
        rx,ry = int(x*sx),int(y*sy)
        rw,rh = int(bw*sx),int(bh*sy)
        cv2.rectangle(vis,(rx,ry),(rx+rw,ry+rh),col,2)
        cv2.putText(vis,f"{i} {lbl}",(rx,ry-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,col,2)

    fascia_status = "fascia:OK" if top_y is not None else "fascia:NONE"
    info = f"{fname_tag}  {fascia_status}  veins:{len(dets)}"
    cv2.putText(vis, info, (6,22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)
    return vis


# ── Extract test frames ───────────────────────────────────────────────────────
def extract_frames(video_path, n=4):
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for pct in np.linspace(0.15, 0.85, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
        ret, f = cap.read()
        if ret:
            frames.append((int(total * pct), f))
    cap.release()
    return frames


# ── Main test loop ────────────────────────────────────────────────────────────
_seen = set()
videos = []
for p in sorted(DATA_DIR.glob('*')):
    if p.suffix.lower() in ('.mp4', '.avi', '.mov') and p.stem not in _seen:
        _seen.add(p.stem); videos.append(p)
print(f"\nFound {len(videos)} videos: {[v.name for v in videos]}\n")

for vid in videos:
    print(f"\n{'='*60}")
    print(f"VIDEO: {vid.name}")
    frames = extract_frames(vid, n=3)

    for fi, bgr in frames:
        tag = f"{vid.stem}_f{fi}"
        print(f"\n  --- frame {fi} ---")

        t0 = time.time()
        top_y, bot_y = detect_fascia(bgr)
        print(f"  fascia time: {time.time()-t0:.1f}s  result: {'OK' if top_y is not None else 'NONE'}")

        t0 = time.time()
        dets = detect_veins(bgr)
        print(f"  veins time:  {time.time()-t0:.1f}s  found: {len(dets)}")

        labels = {}
        if groq_client and dets:
            t0 = time.time()
            labels = vlm_classify(bgr, dets, top_y, bot_y)
            print(f"  VLM time:    {time.time()-t0:.1f}s  labels: {labels}")

        vis = draw_result(bgr, top_y, bot_y, dets, labels, tag)
        out_path = OUT_DIR / f"{tag}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"  -> saved {out_path.name}")

print(f"\nAll outputs in {OUT_DIR}")
