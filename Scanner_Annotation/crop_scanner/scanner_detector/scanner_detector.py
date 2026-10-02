import cv2
import queue
import threading
import traceback
import time
from onnxruntime import get_device

from .config import load_config
from .processor.position_memory import PositionMemory
from .processor.person_detector import PersonDetector
from .processor.pose_detector import PoseDetector
from .processor.scanner_locator import ScannerLocator
from .processor.segment_detector import SegmentDetector

from .shared import debug_q

from .visualizer.visualizer import Visualizer


class ScannerDetector:
    def __init__(self, input_queue, output_queue, config_path=''):
        self.config = load_config(config_path)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False
        
        self.sync_barrier = threading.Barrier(4)
        self.exit_event = threading.Event()
        
        self.inference_thread = threading.Thread(target=self.run, daemon=True)
    
    def start(self):
        print("ScannerDetection thread started.")
        self.running = True
        self.inference_thread.start()
    
    def stop(self):
        self.running = False
        self.exit_event.set()
        self.inference_thread.join()
    
    def run(self):
        # Detect if GPU set up for onnxruntime.
        if get_device() == 'GPU':
            device = 'cuda'
        else:
            device = 'cpu'
        
        # Initialize result collector.
        position_memory = PositionMemory(device)

        # Initialize different stages of the pipeline.
        person_detector = PersonDetector(position_memory, self.sync_barrier, self.exit_event, device=device)
        pose_detector = PoseDetector(position_memory, self.sync_barrier, self.exit_event, device=device)
        scanner_locator = ScannerLocator(position_memory, self.sync_barrier, self.exit_event, device=device)
        # segment_detector = SegmentDetector(position_memory, self.sync_barrier, self.exit_event)
        
        pipeline_stages = [person_detector, pose_detector, scanner_locator]
        for stage in pipeline_stages: stage.start()
        
        # Main loop that fetches frames from input queue and passes to each stage.
        while self.running and not self.exit_event.is_set():
            try:
                try:
                    frame = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Convert frame to RGB channel and submit to the stages.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for stage in pipeline_stages: stage.input_queue.put_nowait(rgb_frame)
                self.sync_barrier.wait()  # wait for all stages to finish
                # Parse results and return to output queue.
                try:
                    scanner_frame = debug_q.get_nowait()
                except queue.Empty:
                    results = None
                else:
                    results = {'scanner_frame': scanner_frame}
                try:
                    self.output_queue.put_nowait(results)
                except queue.Full:
                    pass
                
            except threading.BrokenBarrierError:
                break
            except Exception:
                print("Error: Runtime error on inference thread")
                traceback.print_exc()
                self.running = False
                print("run set")
                self.exit_event.set()
                self.sync_barrier.abort()
                break
        
        for stage in pipeline_stages: stage.worker_thread.join()
        print("ScannerDetection thread closed.")