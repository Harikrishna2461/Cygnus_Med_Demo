import json

annotation_file = "full_leg_level_annotation.json"

with open(annotation_file, 'r') as f:
    data = json.load(f)

left_points = {int(k): v for k, v in data['left'].items()}
sorted_frames = sorted(left_points.keys())

print(f"Current density: {len(left_points)} points")

densified = {}

for i in range(len(sorted_frames) - 1):
    frame_a = sorted_frames[i]
    frame_b = sorted_frames[i + 1]
    level_a = left_points[frame_a]
    level_b = left_points[frame_b]

    densified[frame_a] = level_a

    frames_between = frame_b - frame_a
    if frames_between > 30:
        steps = frames_between // 30
        for step in range(1, steps):
            inter_frame = frame_a + (frame_b - frame_a) * step // steps
            inter_level = level_a + (level_b - level_a) * step / steps
            inter_level = round(inter_level, 2)
            densified[inter_frame] = inter_level

densified[sorted_frames[-1]] = left_points[sorted_frames[-1]]

densified_sorted = {k: densified[k] for k in sorted(densified.keys())}

output = {
    "left": densified_sorted,
    "right": {}
}

with open(annotation_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Refined density: {len(densified_sorted)} points (from {len(left_points)})")
print(f"Interpolated {len(densified_sorted) - len(left_points)} additional points")
