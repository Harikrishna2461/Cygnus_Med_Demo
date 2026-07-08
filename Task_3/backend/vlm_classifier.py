import base64
import json
import re
import time
from groq import Groq

GROQ_API_KEY = ""
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

_client = None


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = (
    "You are a medical image analysis AI. "
    "You output ONLY raw JSON — no markdown fences, no reasoning steps, no extra text."
)


def _build_user_prompt(circles: list[dict]) -> str:
    pos_lines = "\n".join(
        f"  Circle {i}: center at pixel ({c['center_x']}, {c['center_y']})"
        for i, c in enumerate(circles)
    )
    return (
        "BACKGROUND — what you are looking at:\n"
        "This is a B-mode (greyscale) ultrasound cross-section of a human leg. "
        "Tissue appears in shades of grey; veins appear as dark (near-black) oval or round blobs "
        "because blood reflects very little ultrasound. "
        "The FASCIA LAYER is a sheet of dense connective tissue visible as a bright (hyperechoic) "
        "horizontal band. It is explicitly marked with TWO YELLOW HORIZONTAL LINES: "
        "the region between them is the fascial compartment. "
        "Above the upper yellow line = subcutaneous fat (superficial zone). "
        "Below the lower yellow line = deep muscle zone.\n"
        "\n"
        f"There are {len(circles)} vein(s) already detected and marked with green circles "
        "at these exact pixel positions (x = pixels from left, y = pixels from top, y=0 is top):\n"
        f"{pos_lines}\n"
        "\n"
        "TASK — for each circle index, look at where that pixel coordinate sits in the image "
        "relative to the two yellow fascia lines and classify it:\n"
        "  N3 = center ABOVE upper yellow line  (superficial / subcutaneous vein)\n"
        "  N2 = center BETWEEN the two yellow lines  (GSV / fascial compartment)\n"
        "  N1 = center BELOW lower yellow line  (deep vein, inside muscle)\n"
        "\n"
        "Reply with ONLY this JSON (one entry per circle, same order as above):\n"
        '{"veins":[{"index":0,"classification":"N1 or N2 or N3"},{"index":1,"classification":"..."}]}\n'
        'If no circles: {"veins":[]}'
    )


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    clean = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    candidates = re.findall(r'\{[^{}]*"veins"[^{}]*\[[^\[\]]*\][^{}]*\}', text, re.DOTALL)
    if candidates:
        try:
            return json.loads(candidates[-1])
        except json.JSONDecodeError:
            pass

    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return None


def classify_frame(jpeg_bytes: bytes, circles: list[dict]) -> list[dict]:
    """
    Classify pre-detected vein circles.
    circles: [{center_x, center_y, radius}] — exact positions from OpenCV
    Returns [{center_x, center_y, radius, classification}] at those same positions.
    """
    if not circles:
        return []

    b64         = base64.b64encode(jpeg_bytes).decode()
    client      = get_client()
    user_prompt = _build_user_prompt(circles)

    for attempt in range(2):
        text = ""
        try:
            resp = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": user_prompt},
                        ],
                    },
                ],
                temperature=0.0,
                max_tokens=400,
                response_format={"type": "json_object"},
            )

            text = resp.choices[0].message.content.strip()
            data = _extract_json(text)

            if data is None:
                print(f"[VLM] No JSON found (attempt {attempt+1}): {text[:200]}")
                continue

            raw    = data.get("veins", [])
            result = []
            for item in raw:
                idx = int(item.get("index", -1))
                cls = str(item.get("classification", "")).strip().upper()
                if cls not in ("N1", "N2", "N3"):
                    continue
                if 0 <= idx < len(circles):
                    c = circles[idx]
                    result.append({
                        "center_x":       c["center_x"],
                        "center_y":       c["center_y"],
                        "radius":         c["radius"],
                        "classification": cls,
                    })
            return result

        except json.JSONDecodeError as e:
            print(f"[VLM] JSON decode error (attempt {attempt+1}): {e} | text: {text[:200]}")
        except Exception as e:
            print(f"[VLM] API error (attempt {attempt+1}): {e}")
            if attempt == 0:
                time.sleep(1.5)

    return []