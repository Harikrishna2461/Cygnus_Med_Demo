"""
LangGraph ReAct agent for ultrasound scan-area ROI detection.

The agent (Qwen text model via Groq) reasons about the video by calling
image-analysis tools that return numerical statistics — no vision model needed.

Flow:
  1. get_video_metadata   -> know coordinate space
  2. run_cv_detection     -> get CV starting estimate
  3. get_spatial_profile  -> understand content distribution across full frame
  4. analyze_region / probe_boundary (as many calls as needed) -> refine edges
  5. Output FINAL_ROI JSON in last message -> pipeline extracts it
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from agent_tools import ALL_TOOLS

# ── Config ────────────────────────────────────────────────────────────────────

GROQ_API_KEY = ""
GROQ_MODEL   = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = """You are an ultrasound scan-area detection agent.

GOAL: Find bounding box (x1, y1, x2, y2) of ONLY the scan image — exclude UI chrome
(parameter panels, depth rulers, logos, gain bars, sidebars).

KEY METRIC: texture_density (fraction of pixels with significant local texture, 0-1)
- SCAN CONTENT or MIXED: include in scan area
- UI PANEL / BACKGROUND / COLOURED UI CHROME: exclude

MANDATORY 5-STEP WORKFLOW — follow exactly, no deviations:

STEP 1: get_video_metadata  (learn width x height)
STEP 2: run_cv_detection    (get cv_roi = [x1,y1,x2,y2])
STEP 3: get_spatial_profile (read col_profile; note which column ranges are SCAN CONTENT)

  After step 3, build your CANDIDATE ROI:
  - y1, y2: use cv_roi values (CV ensemble is reliable for vertical extent)
  - x1: find the first column range in col_profile classified SCAN CONTENT or MIXED; use its start pixel
  - x2: find the last column range in col_profile classified SCAN CONTENT or MIXED; use its end pixel

STEP 4: analyze_region on the full candidate ROI [x1, y1, x2, y2]
  - If classification is SCAN CONTENT or MIXED: good, proceed to step 5
  - If not: set x1/x2 back to cv_roi x1/x2 as fallback

STEP 5: analyze_region on a 30-px strip at the right edge [x2-30, y1, x2, y2]
  - If UI PANEL or BACKGROUND: shrink x2 by moving left to the previous col_profile boundary
  - If SCAN CONTENT or MIXED: keep x2

Then output FINAL_ROI immediately — do NOT call any more tools.

OUTPUT (last thing you write, nothing after):
FINAL_ROI: {"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>}

RULES:
- NEVER call scan_boundary
- NEVER binary-search pixel by pixel — read col_profile boundaries directly
- MIXED = scan content, always include it
- Make at most 5 tool calls, then output FINAL_ROI
"""

# ── Agent factory ─────────────────────────────────────────────────────────────

def _build_agent():
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=4096,
        # Disable Qwen3 extended thinking so the model emits visible text.
        # With thinking on, the final message is empty (reasoning is internal).
        reasoning_effort="none",
    )
    return create_react_agent(
        llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
    )


# Lazily build one agent and reuse
_agent = None

def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


# ── ROI extraction from final message ─────────────────────────────────────────

def _extract_roi(text: str, w: int, h: int) -> Optional[tuple]:
    """Parse FINAL_ROI or any JSON {x1,y1,x2,y2} from the agent's last message."""
    # Preferred: explicit FINAL_ROI marker
    m = re.search(r'FINAL_ROI\s*[:\-]\s*(\{[^}]+\})', text, re.IGNORECASE)
    if m:
        try:
            d = json.loads(m.group(1))
            return _validate(d, w, h)
        except Exception:
            pass

    # Fallback: any JSON block with x1/y1/x2/y2 keys
    for block in re.findall(r'\{[^{}]+\}', text):
        try:
            d = json.loads(block)
            roi = _validate(d, w, h)
            if roi:
                return roi
        except Exception:
            pass

    return None


def _validate(d: dict, w: int, h: int) -> Optional[tuple]:
    try:
        x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
    except (KeyError, ValueError, TypeError):
        return None
    x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
    if x2 > x1 + 10 and y2 > y1 + 10:
        return (x1, y1, x2, y2)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def _roi_from_tool_results(messages: list, w: int, h: int) -> Optional[tuple]:
    """
    Fallback: reconstruct an ROI from the analyse_region / probe_boundary tool
    results collected during the agent's reasoning loop.

    We gather every region that was classified as SCAN CONTENT and take the
    union bounding box — a reasonable approximation when the model's final
    text message is empty (e.g., Qwen3 extended-thinking leftover).
    """
    import json as _json

    scan_boxes: list[tuple] = []
    for msg in messages:
        # Tool result messages have a .name attribute
        if not (hasattr(msg, "name") and msg.name in ("analyze_region", "probe_boundary")):
            continue
        try:
            data = _json.loads(msg.content)
        except Exception:
            continue

        # analyze_region result
        if "region" in data and "SCAN CONTENT" in data.get("interpretation", ""):
            r = data["region"]
            scan_boxes.append((r[0], r[1], r[2], r[3]))

    if not scan_boxes:
        return None

    # Union of all scan boxes
    ux1 = min(b[0] for b in scan_boxes)
    uy1 = min(b[1] for b in scan_boxes)
    ux2 = max(b[2] for b in scan_boxes)
    uy2 = max(b[3] for b in scan_boxes)
    ux1, ux2 = max(0, ux1), min(w, ux2)
    uy1, uy2 = max(0, uy1), min(h, uy2)
    if ux2 > ux1 + 10 and uy2 > uy1 + 10:
        return (ux1, uy1, ux2, uy2)
    return None


def detect_roi(
    video_path: str,
    status_callback=None,
) -> Optional[tuple]:
    """
    Run the LangGraph ROI detection agent on video_path.
    Returns (x1, y1, x2, y2) or None if detection failed.
    """
    def _status(msg: str):
        print(f"[Agent] {msg}")
        if status_callback:
            status_callback(msg)

    from frame_sampler import get_video_info
    info = get_video_info(video_path)
    w, h = info["width"], info["height"]

    _status(f"Starting ROI agent on {w}x{h} video...")

    agent = _get_agent()

    user_msg = (
        f"Detect the ultrasound scan area in this video file:\n"
        f"  path: {video_path}\n\n"
        "Follow the mandatory 5-step workflow from the system prompt exactly. "
        "Make at most 5 tool calls, then output:\n"
        'FINAL_ROI: {"x1": INT, "y1": INT, "x2": INT, "y2": INT}'
    )

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config={"recursion_limit": 50},
        )
    except Exception as e:
        _status(f"Agent error: {e}")
        return None

    messages = result["messages"]
    n_tool_calls = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)
    _status(f"Agent made {n_tool_calls} tool call(s)")

    # 1. Try explicit FINAL_ROI in the last AI message
    last_msg = messages[-1]
    content  = last_msg.content if hasattr(last_msg, "content") else ""
    if content:
        _status(f"Agent final message: {content[:300]}")
        roi = _extract_roi(content, w, h)
        if roi:
            _status(f"Extracted ROI from final message: {roi}")
            return roi

    # 2. Search ALL AI messages for a FINAL_ROI marker
    all_ai_text = " ".join(
        str(m.content) for m in messages
        if type(m).__name__ == "AIMessage" and m.content
    )
    roi = _extract_roi(all_ai_text, w, h)
    if roi:
        _status(f"Extracted ROI from message history: {roi}")
        return roi

    # 3. Reconstruct from tool results (Qwen3 thinking-mode fallback)
    _status("Final message empty — reconstructing ROI from tool call results...")
    roi = _roi_from_tool_results(messages, w, h)
    if roi:
        _status(f"Reconstructed ROI from tool results: {roi}")
        return roi

    _status("All extraction methods failed")
    return None
