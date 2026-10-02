import threading
import traceback
import queue
import numpy as np
import json

from .position_memory import PositionMemory
from .utils import calculate_iou
from ..shared import test_q


class PersonDetectorGT:
    """
    Wrapper for thread detecting bounding boxes of people.
    Pass RGB frames for processing into `input_queue`.
    """
    def __init__(self, position_memory: PositionMemory, barrier, exit_event, filename):
        """
        Args:
            position_memory (PositionMemory): PositionMemory to read/update algorithm results.
            barrier (threading.Barrier): Barrier to stop the loop until other stages are done.
            exit_event (threading.Event): Control for thread to keep running or exit.
            device (str, optional): Device to load inference model. Defaults to 'cuda'.
            backend (str, optional): Backend framework for inference model. Defaults to 'onnxruntime'.
        """
        self.input_queue = queue.Queue(maxsize=1)
        self.position_memory = position_memory
        self.barrier = barrier
        self.exit_event = exit_event
        self.worker_thread = threading.Thread(target=self.run, daemon=True)
        self.gt = self.load_gt(filename)
        
    def load_gt(self, filename):
        bbox_json_file = f'test_result/{filename}_bbox_result.json'
        with open(bbox_json_file, 'r', encoding='utf-8') as json_file:
            gt = json.load(json_file)
        return gt
    
    def start(self):
        self.worker_thread.start()
        print("    PersonDetector thread started.")
    
    def stop(self):
        self.exit_event.set()
        self.worker_thread.join()
        
    def run(self):
        while not self.exit_event.is_set():
            try:
                try:
                    frame_id = test_q.get(timeout=1)
                    frame = self.input_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Load GT bboxes and return.
                found_bboxes = self.gt[str(frame_id)]['bboxes']
                if len(found_bboxes) == 2:  # updat only if 2 boxes
                    self.position_memory.update_bboxes(found_bboxes)
                else:
                    print("bad gt bbox")
                
                self.barrier.wait()
                
            except threading.BrokenBarrierError:
                break
            except Exception:
                print(f"ERROR in PersonDetector thread:")
                traceback.print_exc()
                self.exit_event.set()
                self.barrier.abort()
                break
        print("    PersonDetector thread closed.")