import cv2, os

out_dir = r'c:\Users\Krish\Downloads\Task_3\frames_preview'
os.makedirs(out_dir, exist_ok=True)

videos = {
    'raw':    r'c:\Users\Krish\Downloads\Task_3\Data\0 - Raw videos',
    'proc':   r'c:\Users\Krish\Downloads\Task_3\Data\1 - Videos',
    'ann':    r'c:\Users\Krish\Downloads\Task_3\Data\2 - Annotated videos',
    'simple': r'c:\Users\Krish\Downloads\Task_3\Data\3 - Simple Annotated videos',
}

clips = ['sample_data.mp4', '202207111318_38-Perf.mp4', '202207191643_00-Moving.mp4', 'Knee-Ankle-Seg1.mp4']

for clip in clips:
    for tag, folder in videos.items():
        path = os.path.join(folder, clip)
        if not os.path.exists(path):
            path = os.path.join(folder, clip.replace('.mp4', '.MP4'))
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f'{tag}/{clip}: {w}x{h}, {fps:.1f}fps, {total} frames, {total/fps:.1f}s')
        for pct in [0.1, 0.5, 0.8]:
            idx = int(total * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                base = clip.replace('.mp4', '').replace('.MP4', '')
                name = f'{base}_{tag}_f{idx}.jpg'
                cv2.imwrite(os.path.join(out_dir, name), frame)
        cap.release()
