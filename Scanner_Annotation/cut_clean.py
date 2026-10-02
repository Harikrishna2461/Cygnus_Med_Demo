import subprocess, cv2, imageio_ffmpeg, os

input_video = r'c:\Users\Krish\Desktop\scanner-annotation\crop_scanner\result_merged\full_scanner_0_to_11.mp4'
output_video = r'c:\Users\Krish\Desktop\scanner-annotation\crop_scanner\result_merged\full_scanner_0_to_11_clean.mp4'

# Remove (1-indexed): 1-1820, 2682-3346, 10330-10818
# Keep (0-indexed, end_frame is exclusive for trim filter):
#   1820 to 2681  (1-indexed 1821-2681, count=861)
#   3346 to 10329 (1-indexed 3347-10329, count=6983)
# Total expected: 7844 frames

keep_ranges = [
    (1820, 2681),   # 0-indexed inclusive
    (3346, 10329),
]

filter_parts = []
concat_inputs = []
for idx, (s, e) in enumerate(keep_ranges):
    # trim end_frame is exclusive
    filter_parts.append(f"[0:v]trim=start_frame={s}:end_frame={e+1},setpts=PTS-STARTPTS[v{idx}]")
    concat_inputs.append(f"[v{idx}]")

filter_complex = ';'.join(filter_parts) + ';' + ''.join(concat_inputs) + f"concat=n={len(keep_ranges)}:v=1:a=0[out]"

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [
    ffmpeg, '-y', '-i', input_video,
    '-filter_complex', filter_complex,
    '-map', '[out]',
    '-c:v', 'libx264', '-crf', '0', '-preset', 'ultrafast',
    output_video
]

print("Running ffmpeg...")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    cap = cv2.VideoCapture(output_video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Done. Output frame count: {n} (expected 7844)")
else:
    print("FAILED")
    print(result.stderr[-3000:])
