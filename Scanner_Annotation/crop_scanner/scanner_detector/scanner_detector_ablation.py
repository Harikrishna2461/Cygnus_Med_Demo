import cv2
import queue
import threading
import traceback
import time
from onnxruntime import get_device

from .config import load_config
from .processor.position_memory import PositionMemory
from .processor.person_detector import PersonDetector
from .processor.person_detector_gt import PersonDetectorGT
from .processor.pose_detector import PoseDetector
from .processor.scanner_locator import ScannerLocator
from .processor.scanner_locator_gt import ScannerLocatorGT
from .processor.segment_detector import SegmentDetector


from .visualizer.visualizer import Visualizer


class ScannerDetectorAblation:
    def __init__(self, input_queue, output_queue, filename, config_path=''):
        self.config = load_config(config_path)
        self.input_queue = input_queue
        self.output_queue = output_queue
        
        self.filename = filename
        
        self.running = False
        
        self.sync_barrier = threading.Barrier(5)
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
        # person_detector = PersonDetector(position_memory, self.sync_barrier, self.exit_event, device=device)
        person_detector = PersonDetectorGT(position_memory, self.sync_barrier, self.exit_event, self.filename)
        pose_detector = PoseDetector(position_memory, self.sync_barrier, self.exit_event, device=device)
        scanner_locator = ScannerLocator(position_memory, self.sync_barrier, self.exit_event, device=device)
        # scanner_locator = ScannerLocatorGT(position_memory, self.sync_barrier, self.exit_event, self.filename)
        segment_detector = SegmentDetector(position_memory, self.sync_barrier, self.exit_event)
        
        pipeline_stages = [person_detector, pose_detector, scanner_locator, segment_detector]
        for stage in pipeline_stages: stage.start()
        
        # Visualizer for results if needed.
        visualizer = Visualizer(config=self.config['VISUALIZER'], position_memory=position_memory)
        
        # Track FPS just for evaluation.
        last_fps, accum_frames, accum_time = 0, 0, 0
        curr_time = time.perf_counter()
        
        # Main loop that fetches frames from input queue and passes to each stage.
        while self.running and not self.exit_event.is_set():
            try:
                try:
                    frame = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                start_time = time.perf_counter()
                
                # Convert frame to RGB channel and submit to the stages.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                for stage in pipeline_stages: stage.input_queue.put_nowait(rgb_frame)
                
                self.sync_barrier.wait()  # wait for all stages to finish
                
                end_time = time.perf_counter()
                
                # Parse results and return to output queue.
                results = position_memory.get_results()
                            
                # Track FPS for evaluation.
                accum_frames += 1
                accum_time += end_time - start_time
                if end_time - curr_time > 0.5:
                    last_fps = accum_frames / accum_time
                    accum_frames, accum_time = 0, 0
                    curr_time = end_time
                results['fps'] = last_fps
                
                if self.config['VISUALIZER']['draw']:
                    results = results | visualizer.get_drawn_frames(frame)
                
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