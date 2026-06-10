"""
Foundation pipeline quality test — VLM combined approach.
Tests what the actual pipeline does: VLM (Groq) detects fascia + veins + N1/N2/N3
in one shot with few-shot reference images from 3-Simple_Annotated_Videos.

Run: python test_foundation_vlm.py
Outputs annotated frames to test_output_vlm/
"""
import sys, os, cv2, json, re, base64, time
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

IMG_SIZE   = 256
HERE       = Path(__file__).parent
DATA_ROOT  = HERE / 'vein_detection_task_3_training'
DATA_DIR   = DATA_ROOT / 'Data' / '1-Videos'
ANNO_DIR   = DATA_ROOT / 'Data' / '3-Simple_Annotated_Videos'
OUT_DIR    = HERE / 'test_output_vlm'
OUT_DIR.mkdir(exist_ok=True)

GROQ_KEY = os.environ.get('GROQ_KEY', '')

# ── Groq ──────────────────────────────────────────────────────────────────────
from groq import Groq
groq_client = Groq(api_key=GROQ_KEY)
print("Groq client ready.")

# ── GDINO + SAM (for fascia only — veins via VLM) ─────────────────────────────
print("Loading Grounding DINO + SAM...")
import torch
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor)
from PIL import Image as PILImage

gdino_proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny").eval()
sam_model   = SamModel.from_pretrained("facebook/sam-vit-base").eval()
sam_proc    = SamProcessor.from_pretrained("facebook/sam-vit-base")
print("GDINO + SAM ready.")


# ── Few-shot loader (mirrors _load_few_shot in app.py) ───────────────────────
FEW_SHOT_PROMPT = (
    "You are analysing B-mode ultrasound images.\n"
    "The annotated reference images you already saw show:\n"
    "  • YELLOW lines  = fascia band (bright hyperechoic layer between fat and muscle)\n"
    "  • Coloured numbered boxes = veins (dark hypoechoic oval/circular structures)\n"
    "  • Labels: N3=above fascia (superficial), N2=at/near fascia (GSV), N1=below fascia (deep)\n\n"
    "Now analyse this new unannotated frame using the same anatomy.\n"
    "Find: (1) the fascia band row, (2) any dark oval vein structures.\n"
    "Reply ONLY with valid JSON — no extra text:\n"
    "{\"fascia_y\":<integer 0-255 for fascia centre row in this 256x256 image, or null>,"
    "\"veins\":[{\"x\":<int>,\"y\":<int>,\"w\":<int>,\"h\":<int>,\"label\":\"N2\"}]}\n"
    "All pixel coords are in 256x256 space (0,0 = top-left)."
)

def load_few_shot():
    few_shot_b64 = []
    for fname in ['202207191643_00-Moving.mp4', '202207111318_38-Perf.mp4', 'sample_data.mp4']:
        p = ANNO_DIR / fname
        if not p.exists():
            continue
        cap   = cv2.VideoCapture(str(p))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 5:
            cap.release(); continue
        for pct in [0.20, 0.50]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * pct)))
            ret, frm = cap.read()
            if ret:
                small = cv2.resize(frm, (IMG_SIZE, IMG_SIZE))
                _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 85])
                few_shot_b64.append(base64.b64encode(buf.tobytes()).decode())
        cap.release()
        if len(few_shot_b64) >= 4:
            break
    print(f"Few-shot: {len(few_shot_b64)} reference frames loaded from {ANNO_DIR.name}")
    return few_shot_b64

few_shot_b64 = load_few_shot()

# Pre-build context messages (same structure as app.py _load_few_shot)
ref_content = [{"type": "text", "text": (
    "These are annotated B-mode ultrasound reference images.\n"
    "YELLOW lines = fascia band (bright hyperechoic layer between fat and muscle).\n"
    "Coloured numbered boxes = veins (dark anechoic/hypoechoic ovals). "
    "Veins can be LARGE (filling 20-50% of the image) or small.\n"
    "Labels: N3=above fascia (superficial), N2=at/near fascia (GSV), N1=below fascia (deep)."
)}]
for b64 in few_shot_b64[:4]:
    ref_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

vlm_context_msgs = [
    {"role": "user",      "content": ref_content},
    {"role": "assistant", "content": (
        "Understood. I can see the annotated reference frames: yellow fascia lines "
        "and numbered vein boxes (including large dark oval vessels) with N1/N2/N3 labels. "
        "Ready to analyse new frames."
    )},
]
print(f"VLM context pre-built with {len(few_shot_b64)} reference images.")


# ── Combined VLM detection (mirrors vlm_detect_classify in app.py) ───────────
def vlm_combined_detect(bgr):
    """Groq VLM: detect fascia_y + veins + N1/N2/N3 in one shot."""
    vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    _, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
    cur_b64 = base64.b64encode(buf.tobytes()).decode()

    query_msg = {"role": "user", "content": [
        {"type": "text",      "text": "Analyse this new frame:"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cur_b64}"}},
        {"type": "text",      "text": FEW_SHOT_PROMPT},
    ]}
    messages = vlm_context_msgs + [query_msg]

    try:
        resp = groq_client.chat.completions.create(
            model='meta-llama/llama-4-scout-17b-16e-instruct',
            messages=messages,
            max_tokens=500, temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        m   = re.search(r'\{.*\}', raw, re.DOTALL)
        if not m:
            print(f"  VLM: no JSON in response: {raw[:100]}")
            return None, []
        data = json.loads(m.group())

        fascia_y = data.get('fascia_y')
        if fascia_y is not None:
            fascia_y = int(np.clip(int(fascia_y), 0, IMG_SIZE - 1))

        dets = []
        for v in data.get('veins', []):
            try:
                x = int(np.clip(int(v['x']), 0, IMG_SIZE - 1))
                y = int(np.clip(int(v['y']), 0, IMG_SIZE - 1))
                w = max(4, int(np.clip(int(v['w']), 1, IMG_SIZE - x)))
                h = max(4, int(np.clip(int(v['h']), 1, IMG_SIZE - y)))
                label = v.get('label', 'N2')
                if label not in ('N1', 'N2', 'N3'):
                    label = 'N2'
                dets.append({'bbox': (x,y,w,h), 'centroid': (x+w/2, y+h/2),
                             'area': w*h, 'vlm_label': label})
            except (KeyError, ValueError, TypeError):
                continue
        return fascia_y, dets
    except Exception as e:
        print(f"  VLM error: {e}")
        return None, []


# ── SAM helpers for fascia refinement ────────────────────────────────────────
def gdino_fascia_box(bgr):
    h0, w0 = bgr.shape[:2]
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    inputs = gdino_proc(images=pil,
                        text="fascia . bright horizontal hyperechoic band . connective tissue layer",
                        return_tensors="pt")
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    try:
        results = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids, target_sizes=[(h0, w0)])[0]
        keep = results['scores'].cpu() >= 0.20
        boxes = results['boxes'][keep].cpu().numpy()
    except TypeError:
        results = gdino_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=0.20, text_threshold=0.16, target_sizes=[(h0, w0)])[0]
        boxes = results['boxes'].cpu().numpy()

    best_box, best_score = None, -1.0
    for (x1, y1, x2, y2) in boxes:
        cy_norm = ((y1 + y2) / 2) / h0
        if cy_norm < 0.10 or cy_norm > 0.80:
            continue
        bw, bh = x2 - x1, max(y2 - y1, 1)
        score = (bw / w0) * (bw / bh)
        if score > best_score:
            best_score = score; best_box = (x1, y1, x2, y2)
    return best_box


def sam_fascia_from_box(bgr, box_xyxy):
    pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    inputs = sam_proc(pil, input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
    with torch.no_grad():
        outputs = sam_model(**inputs)
    masks = sam_proc.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())
    iou_scores = outputs.iou_scores[0, 0].cpu().numpy()
    best_mask  = masks[0][0, int(np.argmax(iou_scores))].numpy().astype(np.uint8)
    mask_r = cv2.resize(best_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    top_out = np.full(IMG_SIZE, np.nan, np.float32)
    bot_out = np.full(IMG_SIZE, np.nan, np.float32)
    for x in range(IMG_SIZE):
        ys = np.where(mask_r[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        ctr = float(np.median(ys))
        top_out[x] = max(0, ctr - 8)
        bot_out[x] = min(IMG_SIZE - 1, ctr + 8)

    valid = ~np.isnan(top_out)
    if valid.sum() < 10:
        return None, None

    xs      = np.where(valid)[0].astype(np.float32)
    x_min   = int(xs.min()); x_max = int(xs.max())
    scan_xs = np.arange(x_min, x_max + 1, dtype=np.float32)

    # IQR rejection
    tv = top_out[valid]
    q25, q75 = float(np.percentile(tv, 25)), float(np.percentile(tv, 75))
    iqr = q75 - q25 if q75 > q25 else 6.0
    top_out[valid] = np.clip(tv, q25 - 2*iqr, q75 + 2*iqr)
    bot_out[valid] = np.clip(bot_out[valid], q25 - 2*iqr + 16, q75 + 2*iqr + 16)

    top_out[x_min:x_max+1] = uniform_filter1d(
        np.interp(scan_xs, xs, top_out[valid]), 60).astype(np.float32)
    bot_out[x_min:x_max+1] = uniform_filter1d(
        np.interp(scan_xs, xs, bot_out[valid]), 60).astype(np.float32)
    return top_out, bot_out


def sam_fascia_from_vlm_y(bgr, fascia_y):
    h0, w0 = bgr.shape[:2]
    search = 30
    y1_box = max(0,  int((fascia_y - search) * h0 / IMG_SIZE))
    y2_box = min(h0, int((fascia_y + search) * h0 / IMG_SIZE))
    return sam_fascia_from_box(bgr, (0, y1_box, w0, y2_box))


# ── Drawing ───────────────────────────────────────────────────────────────────
LABEL_COLOR = {'N1':(50,50,220),'N2':(50,200,50),'N3':(0,140,255),'?':(150,150,150)}

def draw_result(bgr, top_y, bot_y, dets, title):
    vis = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE)).copy()
    h, w = vis.shape[:2]

    if top_y is not None:
        xs_v = [x for x in range(IMG_SIZE) if not np.isnan(top_y[x])]
        if xs_v:
            ov = vis.copy()
            pts_t = np.array([[x, int(top_y[x])] for x in xs_v], np.int32)
            pts_b = np.array([[x, int(bot_y[x])] for x in xs_v], np.int32)
            cv2.fillPoly(ov, [np.vstack([pts_t, pts_b[::-1]])], (0,180,180))
            cv2.addWeighted(ov, 0.18, vis, 0.82, 0, vis)
            cv2.polylines(vis, [pts_t], False, (0,255,255), 2)
            cv2.polylines(vis, [pts_b], False, (0,255,255), 2)

    for i, d in enumerate(dets, 1):
        lbl = d.get('vlm_label', d.get('label', '?'))
        col = LABEL_COLOR.get(lbl, (150,150,150))
        x,y,bw,bh = d['bbox']
        cv2.rectangle(vis, (x,y), (x+bw,y+bh), col, 2)
        cv2.putText(vis, f"{i} {lbl}", (x, max(y-4,10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    fascia_txt = "fascia:OK" if top_y is not None else "fascia:NONE"
    cv2.putText(vis, f"{title}  {fascia_txt}  veins:{len(dets)}",
                (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,0), 1)
    return vis


# ── Test loop ─────────────────────────────────────────────────────────────────
_seen = set()
videos = []
for p in sorted(DATA_DIR.glob('*')):
    if p.suffix.lower() in ('.mp4','.avi','.mov','.MP4') and p.stem not in _seen:
        _seen.add(p.stem); videos.append(p)
print(f"\nFound {len(videos)} videos: {[v.name for v in videos]}\n")

for vid in videos:
    print(f"\n{'='*60}")
    print(f"VIDEO: {vid.name}")
    cap   = cv2.VideoCapture(str(vid))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for pct in np.linspace(0.15, 0.85, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * pct))
        ret, f = cap.read()
        if ret:
            frames.append((int(total * pct), f))
    cap.release()

    for fi, bgr in frames:
        tag = f"{vid.stem}_f{fi}"
        print(f"\n  --- frame {fi} ---")

        # ── A: GDINO+SAM fascia
        t0 = time.time()
        box = gdino_fascia_box(bgr)
        if box is not None:
            top_gdino, bot_gdino = sam_fascia_from_box(bgr, box)
        else:
            top_gdino = bot_gdino = None
        print(f"  GDINO+SAM fascia: {'OK' if top_gdino is not None else 'NONE'}  {time.time()-t0:.1f}s")

        # ── B: Combined VLM (fascia + veins + N1/N2/N3)
        t0 = time.time()
        fy, vd = vlm_combined_detect(bgr)
        print(f"  VLM combined: fascia_y={fy}  veins={len(vd)} {[d['vlm_label'] for d in vd]}  {time.time()-t0:.1f}s")

        # SAM-refine VLM fascia_y to curved lines
        top_vlm = bot_vlm = None
        if fy is not None:
            top_vlm, bot_vlm = sam_fascia_from_vlm_y(bgr, fy)

        # ── Best result: prefer VLM fascia when available
        top_best = top_vlm if top_vlm is not None else top_gdino
        bot_best = bot_vlm if bot_vlm is not None else bot_gdino

        # ── Draw side-by-side: GDINO+SAM | VLM | Best
        img_gdino = draw_result(bgr, top_gdino, bot_gdino, [], "GDINO+SAM")
        img_vlm   = draw_result(bgr, top_vlm,   bot_vlm,   vd, "VLM")
        img_best  = draw_result(bgr, top_best,  bot_best,  vd, "BEST")

        combined = np.hstack([img_gdino, img_vlm, img_best])
        out_path = OUT_DIR / f"{tag}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"  -> {out_path.name}")

print(f"\nAll outputs in {OUT_DIR}")
