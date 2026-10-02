import json

annotation_file = "full_leg_level_annotation.json"

with open(annotation_file, 'r') as f:
    data = json.load(f)

def densify_leg(points_dict):
    left_points = {int(k): v for k, v in points_dict.items()}
    sorted_frames = sorted(left_points.keys())

    if not sorted_frames:
        return {}

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
    return {str(k): v for k, v in sorted(densified.items())}

left_densified = densify_leg(data.get('left', {}))
right_densified = densify_leg(data.get('right', {}))

output = {
    "left": left_densified,
    "right": right_densified
}

with open(annotation_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"032Z Densification Complete:")
print(f"  Left: {len(data.get('left', {}))} -> {len(left_densified)} points")
print(f"  Right: {len(data.get('right', {}))} -> {len(right_densified)} points")
