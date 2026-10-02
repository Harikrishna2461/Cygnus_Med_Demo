import os, subprocess, cv2, imageio_ffmpeg

result_dir = r'c:\Users\Krish\Desktop\scanner-annotation\crop_scanner\result'
out_dir = r'c:\Users\Krish\Desktop\scanner-annotation\crop_scanner\result_merged'
concat_list = os.path.join(out_dir, 'concat_list.txt')
output = os.path.join(out_dir, 'full_scanner_0_to_11.mp4')

os.makedirs(out_dir, exist_ok=True)

lines = []
for i in range(12):
    path = os.path.join(result_dir, 'full_scanner_' + str(i) + '.mp4')
    lines.append("file '" + path.replace('\\', '/') + "'")

with open(concat_list, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print("Concat list written:")
print('\n'.join(lines))

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [ffmpeg, '-y', '-f', 'concat', '-safe', '0', '-i', concat_list, '-c', 'copy', output]
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    cap = cv2.VideoCapture(output)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print('Merged OK. Frame count:', n)
else:
    print('FAILED')
    print(result.stderr[-2000:])
