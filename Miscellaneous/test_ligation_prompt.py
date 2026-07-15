import sys
sys.path.insert(0, 'c:/Users/Krish/Downloads/Cygnus_Med_Demo/Task_1_App/backend')

from shunt_classification_and_ligation_llm import build_ligation_prompt

# Create sample clips for Type 1 shunt
clips = [
    {
        "flow": "EP",
        "fromType": "N1",
        "toType": "N2",
        "posYRatio": 0.05,
        "step": "SFJ-Knee"
    },
    {
        "flow": "RP",
        "fromType": "N2",
        "toType": "N1",
        "posYRatio": 0.13,
        "step": "SFJ-Knee"
    }
]

prompt = build_ligation_prompt("Type 1", clips, "Knowledge: Type 1 requires SFJ ligation", "Right")

with open('c:/Users/Krish/Downloads/prompt_output.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("NEW LIGATION PROMPT:\n")
    f.write("=" * 80 + "\n")
    f.write(prompt)

print("Prompt saved to prompt_output.txt")
