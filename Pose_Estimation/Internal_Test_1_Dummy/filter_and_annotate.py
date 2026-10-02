import json
import os

with open("movement_analysis.json", 'r') as f:
    movements = json.load(f)

DIFF_THRESHOLD = 12.0

significant_movements = [m for m in movements if m['diff_before_after'] > DIFF_THRESHOLD]

print(f"Movements with diff > {DIFF_THRESHOLD}: {len(significant_movements)} out of {len(movements)}")
print("\nTop 30 segments (candidates for annotation):")
sorted_movements = sorted(significant_movements, key=lambda x: x['diff_before_after'], reverse=True)
for i, mov in enumerate(sorted_movements[:30]):
    print(f"{i+1:2d}. Frame {mov['mid_frame']:6d} @ {mov['mid_frame_sec']:8.1f}s | diff={mov['diff_before_after']:6.2f} | dur={mov['duration_sec']:6.1f}s")

print("\n\nCreating annotation template with frame numbers:")
annotation_template = {}
for mov in sorted_movements[:80]:
    frame_num = str(mov['mid_frame'])
    annotation_template[frame_num] = "TBD"

with open("annotation_template.json", 'w') as f:
    json.dump(annotation_template, f, indent=2)

print(f"Template created with {len(annotation_template)} candidate frames")
print("Saved to: annotation_template.json")
