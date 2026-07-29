"""Temporary debug script — trace agent tool calls."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from langchain_core.messages import HumanMessage
from roi_agent import _get_agent, _extract_roi
from frame_sampler import get_video_info

video_path = r"C:\Users\Krish\Downloads\Cygnus_Med_Demo\Task_3\Data\0 - Raw videos\sample_data.mp4"
info = get_video_info(video_path)
w, h = info["width"], info["height"]

agent = _get_agent()
user_msg = (
    f"Detect the ultrasound scan area in: {video_path}\n"
    "Use the tools, then finish with: FINAL_ROI: {\"x1\": INT, \"y1\": INT, \"x2\": INT, \"y2\": INT}"
)

result = agent.invoke(
    {"messages": [HumanMessage(content=user_msg)]},
    config={"recursion_limit": 60},
)

full_text = []
for msg in result["messages"]:
    role = type(msg).__name__
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            args_preview = str(list(tc["args"].values()))[:80]
            print(f"[AI->TOOL] {tc['name']}  args={args_preview}")
    elif hasattr(msg, "name") and msg.name:
        content = str(msg.content)[:150]
        print(f"[TOOL={msg.name}] {content}")
    elif role == "AIMessage":
        # Dump full raw repr to see thinking blocks / metadata
        raw_content = msg.content
        print(f"[AI MSG] type(content)={type(raw_content).__name__}  len={len(str(raw_content))}")
        print(f"[AI MSG] repr: {repr(raw_content)[:600]}")
        if raw_content:
            full_text.append(str(raw_content))

# Search ALL AI messages for FINAL_ROI
all_ai_text = "\n".join(full_text)
print(f"\n--- All AI text ---\n{all_ai_text[:1000]}")
roi = _extract_roi(all_ai_text, w, h)
print(f"\nExtracted ROI: {roi}")
