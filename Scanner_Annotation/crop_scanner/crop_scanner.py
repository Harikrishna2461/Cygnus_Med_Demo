import cv2
import queue
import numpy as np
import json

from scanner_detector.scanner_detector import ScannerDetector as ScannerDetector


video_list = [
    "full",
]

show_frame = True

def main():
    for video_name in video_list:
        inference_video(video_name)

def inference_video(video_name):
    src_video = f'data/{video_name}.mp4'
    out_video = f'result/{video_name}_scanner'

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NumpyEncoder, self).default(obj)

    cap = cv2.VideoCapture(src_video)
    frame_idx = 0
    video_id = 0

    # Initialize video writer.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video + f'_{video_id}.mp4', fourcc, 30.0, (168, 168))

    input_queue = queue.Queue(maxsize=1)
    output_queue = queue.Queue(maxsize=1)

    detector = ScannerDetector(input_queue, output_queue)
    detector.start()

    is_running = True
    while is_running:
        ret, frame = cap.read()
        if not ret:
            print(f"EOF reached after {frame_idx} frames.")
            detector.stop()
            break

        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            input_queue.put(frame.copy(), timeout=5)
            results = output_queue.get(timeout=5)
        except queue.Empty:
            print("No results")
            break

        if results is None:
            results = {}

        if 'scanner_frame' in results:
            out_frame = results['scanner_frame']
            results.pop('scanner_frame')
            out.write(out_frame)
            if show_frame:
                cv2.imshow('Scanner Frame', out_frame)

        frame_idx += 1
        print(f"Processed {frame_idx} frames...", end='\r')

        if show_frame:
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
        if frame_idx % 1000 == 0:  # new video every 1000 frames
            video_id += 1
            out.release()
            out = cv2.VideoWriter(out_video + f'_{video_id}.mp4', fourcc, 30.0, (168, 168))

        if not detector.running or detector.exit_event.is_set():
            is_running = False

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()