import os
import sys
import uuid
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'stubs'))
sys.path.insert(0, os.path.join(BASE_DIR, 'BiomedParse'))

from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model load
# ---------------------------------------------------------------------------
print("Loading BiomedParse model...")
BIOMEDPARSE_DIR = os.path.join(BASE_DIR, 'BiomedParse')
opt = load_opt_from_config_files([os.path.join(BIOMEDPARSE_DIR, 'configs', 'biomedparse_inference.yaml')])
opt = init_distributed(opt)

LOCAL_WEIGHTS = os.path.join(BASE_DIR, 'pretrained', 'biomedparse_v1.pt')
pretrained_source = LOCAL_WEIGHTS if os.path.exists(LOCAL_WEIGHTS) else 'hf_hub:microsoft/BiomedParse'
print(f"  weights: {pretrained_source}")

model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_source).eval().cuda()
with torch.no_grad():
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"], is_eval=True
    )
print("Model loaded.")

# Image transform matching BiomedParse training (1024x1024 bicubic)
_img_transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.BICUBIC)
])

VEIN_COLOR   = (0, 210, 0)    # green (RGB)
FASCIA_COLOR = (0, 210, 210)  # cyan  (RGB)

# ---------------------------------------------------------------------------
# Reference image/mask paths
# 3 examples per structure, selected from Task_3_Experimental training data
# using has_fascia/has_vein=True rows spread across different videos.
# ---------------------------------------------------------------------------
TASK3_OUT = os.path.join(
    BASE_DIR, '..', 'Task_3_Experimental',
    'vein_detection_task_3_training', 'output'
)

FASCIA_REFS = [
    (
        os.path.join(TASK3_OUT, 'fascia', 'frames', '202207111318_38-Perf', '00050.png'),
        os.path.join(TASK3_OUT, 'fascia', 'masks',  '202207111318_38-Perf', '00050.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'fascia', 'frames', '202207191643_00-Moving', '00598.png'),
        os.path.join(TASK3_OUT, 'fascia', 'masks',  '202207191643_00-Moving', '00598.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'fascia', 'frames', 'sample_data', '00084.png'),
        os.path.join(TASK3_OUT, 'fascia', 'masks',  'sample_data', '00084.png'),
    ),
]

VEIN_REFS = [
    (
        os.path.join(TASK3_OUT, 'vein', 'frames', '202207111318_38-Perf_00004.png'),
        os.path.join(TASK3_OUT, 'vein', 'masks',  '202207111318_38-Perf_00004.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'vein', 'frames', '202207191643_00-Moving_00508.png'),
        os.path.join(TASK3_OUT, 'vein', 'masks',  '202207191643_00-Moving_00508.png'),
    ),
    (
        os.path.join(TASK3_OUT, 'vein', 'frames', 'sample_data_00212.png'),
        os.path.join(TASK3_OUT, 'vein', 'masks',  'sample_data_00212.png'),
    ),
]


# ---------------------------------------------------------------------------
# Visual few-shot: encode reference → infer query
# ---------------------------------------------------------------------------

def _img_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """Resize to 1024x1024 and convert to [C, H, W] CUDA tensor."""
    resized = _img_transform(pil_image)
    arr = np.asarray(resized).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def encode_reference(ref_image_pil: Image.Image, ref_mask_pil: Image.Image) -> dict:
    """
    Run a (reference image, binary mask) pair through BiomedParse's
    spatial→visual referencing pipeline (task='refimg') to produce
    visual query features that can be injected into any query image pass.

    This is BiomedParse's built-in few-shot visual prompting mechanism:
    the model sees ONE annotated example and learns what the target
    structure looks like, then segments it in unseen images.
    """
    img_tensor = _img_to_tensor(ref_image_pil)

    # Mask must be 1024x1024 boolean, shape [1, H, W]
    mask_arr = np.array(ref_mask_pil.resize((1024, 1024), Image.NEAREST))
    mask_bool = torch.from_numpy(mask_arr > 127).cuda().unsqueeze(0)

    data = {
        'image':         img_tensor,
        'spatial_query': {'rand_shape': mask_bool},
        'height':        ref_image_pil.size[1],
        'width':         ref_image_pil.size[0],
    }

    # spatial=True is required for evaluate_referring_image
    model.model.task_switch['spatial']  = True
    model.model.task_switch['visual']   = True
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio']    = False

    visual_features, _ = model.model.evaluate_referring_image([data])
    return visual_features   # dict: visual_query_pos/neg, src_visual_queries/maskings


def vlm_visual_infer(
    query_image_pil: Image.Image,
    visual_features_list: list,
    threshold: float = 0.40,
) -> np.ndarray:
    """
    Segment a query image using a list of pre-encoded visual reference
    feature dicts (one per few-shot example). Takes the pixel-wise max
    probability across all examples (ensemble), then thresholds.

    How it works:
      1. Run query image through BiomedParse decoder with visual_features
         injected in the 'visual' slot → activates the visual attention path.
      2. Compare each mask proposal's caption embedding against the visual
         reference embedding (pred_pvisuals) via cosine similarity.
      3. Best-matching proposal is selected; sigmoid → probability map.
      4. Repeat for each reference; take max across all (best-of-N ensemble).
    """
    img_tensor = _img_to_tensor(query_image_pil)
    W, H = query_image_pil.size  # PIL gives (width, height)

    model.model.task_switch['spatial']   = True
    model.model.task_switch['visual']    = True
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio']     = False

    best_prob = None

    for vis_feat in visual_features_list:
        data = {
            'image':  img_tensor,
            'visual': vis_feat,
            'height': H,
            'width':  W,
        }

        results, _, _ = model.model.evaluate_demo([data])

        # Visual mode returns different keys than text-grounding mode.
        # Prefer pred_masks (object queries); fall back to pred_smasks (spatial).
        if 'pred_masks' in results:
            pred_masks = results['pred_masks'][0]
        elif 'pred_smasks' in results:
            pred_masks = results['pred_smasks'][0]
        else:
            continue

        # --- Select the best mask proposal ---
        # Priority: visual-reference similarity > logit score > mean activation
        vis_pos      = results.get('pred_pvisuals')   # visual ref embedding
        pred_maskemb = results.get('pred_maskembs')   # mask embeddings
        pred_caps    = results.get('pred_captions')   # caption embeddings
        pred_logits  = results.get('pred_logits')

        if vis_pos is not None and vis_pos.numel() > 0:
            vis_q = vis_pos[0, 0]
            vis_q = vis_q / (vis_q.norm() + 1e-7)
            # Use whichever embedding space is available for cosine similarity
            if pred_maskemb is not None:
                emb   = pred_maskemb[0]
            elif pred_caps is not None:
                emb   = pred_caps[0]
            else:
                emb   = None

            if emb is not None:
                emb_n   = emb / (emb.norm(dim=-1, keepdim=True) + 1e-7)
                matched = (emb_n @ vis_q).argmax()
            elif pred_logits is not None:
                matched = pred_logits[0].max(dim=-1)[0].argmax()
            else:
                matched = pred_masks.sigmoid().mean(dim=(-2, -1)).argmax()
        elif pred_logits is not None:
            matched = pred_logits[0].max(dim=-1)[0].argmax()
        else:
            matched = pred_masks.sigmoid().mean(dim=(-2, -1)).argmax()

        prob = F.interpolate(
            pred_masks[matched:matched+1, :, :][None],
            (H, W), mode='bilinear'
        )[0, 0, :H, :W].sigmoid().cpu().numpy()

        best_prob = prob if best_prob is None else np.maximum(best_prob, prob)

    if best_prob is None:
        return np.zeros((H, W), dtype=np.uint8)
    return (best_prob > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pre-compute visual reference features at startup
# ---------------------------------------------------------------------------

def _load_ref_pair(img_path, mask_path):
    img  = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    return img, mask


print("Encoding visual reference examples (few-shot setup)...")
with torch.no_grad():
    FASCIA_VIS_FEATURES = []
    for img_path, msk_path in FASCIA_REFS:
        if os.path.exists(img_path) and os.path.exists(msk_path):
            img, msk = _load_ref_pair(img_path, msk_path)
            FASCIA_VIS_FEATURES.append(encode_reference(img, msk))
        else:
            print(f"  [WARN] missing fascia ref: {img_path}")

    VEIN_VIS_FEATURES = []
    for img_path, msk_path in VEIN_REFS:
        if os.path.exists(img_path) and os.path.exists(msk_path):
            img, msk = _load_ref_pair(img_path, msk_path)
            VEIN_VIS_FEATURES.append(encode_reference(img, msk))
        else:
            print(f"  [WARN] missing vein ref: {img_path}")

print(f"  {len(FASCIA_VIS_FEATURES)} fascia refs, {len(VEIN_VIS_FEATURES)} vein refs encoded.")
print("Ready. Open http://localhost:5000")


# ---------------------------------------------------------------------------
# Classical CV — fallback / enhancement layer
# Runs on every frame and is unioned with the VLM result.
# ---------------------------------------------------------------------------

def cv_detect_vein(gray: np.ndarray) -> np.ndarray:
    """
    Veins = anechoic (very dark), oval/circular blobs.
    Thresholds for truly dark pixels → morphological clean-up →
    filter by circularity and area → keep top-3 largest round blobs.
    """
    h, w = gray.shape
    blurred  = cv2.GaussianBlur(gray, (15, 15), 5)
    mean_val = float(np.mean(blurred))
    std_val  = float(np.std(blurred))

    dark = (blurred < (mean_val - 0.70 * std_val)).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,  9))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k_close)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  k_open)

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (0.008 * h * w < area < 0.25 * h * w):
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity > 0.35:
            candidates.append((area, cnt))

    candidates.sort(key=lambda x: x[0], reverse=True)
    mask = np.zeros((h, w), dtype=np.uint8)
    for _, cnt in candidates[:3]:
        cv2.drawContours(mask, [cnt], -1, 1, cv2.FILLED)
    return mask


def cv_detect_fascia(gray: np.ndarray, vein_mask: np.ndarray) -> np.ndarray:
    """
    Fascia = hyperechoic (bright), thin, wide horizontal/curvilinear lines.
    Skips top 10% / bottom 5% (ultrasound machine border) to avoid false
    positives at the image edges.
    """
    h, w = gray.shape
    top_margin = int(0.10 * h)
    bot_margin = int(0.05 * h)
    roi = gray[top_margin: h - bot_margin, :]

    blurred   = cv2.GaussianBlur(roi, (3, 3), 0)
    edges     = cv2.Canny(blurred, 8, 30)
    h_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (91, 1))
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)
    v_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    connected = cv2.dilate(connected, v_kernel, iterations=1)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_mask = np.zeros_like(roi, dtype=np.uint8)
    for cnt in contours:
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bw > 0.30 * w and bw / (bh + 1e-5) > 8:
            cv2.drawContours(roi_mask, [cnt], -1, 1, 4)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[top_margin: h - bot_margin, :] = roi_mask

    if vein_mask is not None:
        excl = cv2.dilate(vein_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        mask[excl > 0] = 0
    return mask


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(image_rgb: np.ndarray, vein_mask: np.ndarray, fascia_mask: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()

    # Fascia: thin cyan lines (boundary, not region)
    if fascia_mask is not None and fascia_mask.max() > 0:
        cnts, _ = cv2.findContours(fascia_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, FASCIA_COLOR, 3)

    # Vein: 25% fill + crisp green contour (drawn last so it's never hidden)
    if vein_mask is not None and vein_mask.max() > 0:
        fill = out.copy()
        fill[vein_mask > 0] = VEIN_COLOR
        out = cv2.addWeighted(out, 0.75, fill, 0.25, 0)
        cnts, _ = cv2.findContours(vein_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, VEIN_COLOR, 3)

    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    uid  = uuid.uuid4().hex
    input_path  = os.path.join(UPLOAD_DIR, f'{uid}_input.png')
    output_path = os.path.join(OUTPUT_DIR, f'{uid}_output.png')

    image = Image.open(file.stream).convert('RGB')
    image.save(input_path)

    image_np = np.array(image)
    gray     = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    with torch.no_grad():
        # --- Vein: visual few-shot (ensemble of 3 refs) + CV union ---
        vlm_vein  = vlm_visual_infer(image, VEIN_VIS_FEATURES,   threshold=0.40)
        cv_vein   = cv_detect_vein(gray)
        vein_mask = np.clip(vlm_vein.astype(np.int32) + cv_vein.astype(np.int32), 0, 1).astype(np.uint8)

        # --- Fascia: visual few-shot (ensemble of 3 refs) + CV union ---
        vlm_fascia  = vlm_visual_infer(image, FASCIA_VIS_FEATURES, threshold=0.40)
        cv_fascia   = cv_detect_fascia(gray, vein_mask)
        fascia_mask = np.clip(vlm_fascia.astype(np.int32) + cv_fascia.astype(np.int32), 0, 1).astype(np.uint8)

    annotated = annotate(image_np, vein_mask, fascia_mask)
    Image.fromarray(annotated).save(output_path)
    return send_file(output_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
