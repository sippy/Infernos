from typing import Optional, List
from queue import Queue, Empty as QueueEmpty
from abc import ABC, abstractmethod

from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun

class InfernBatchedWorker(InfernWrkThread, ABC):
    max_batch_size: int
    inf_queue: Queue[Optional[object]]
    def __init__(self):
        super().__init__()
        self.inf_queue = Queue()

    def infer(self, wi:object):
        self.inf_queue.put(wi)

    def next_batch(self) -> List[object]:
        wis = []
        while len(wis) < self.max_batch_size:
            if len(wis) == 0:
                wi = self.inf_queue.get()
            else:
                try: wi = self.inf_queue.get_nowait()
                except QueueEmpty: break
            if wi is None:
                return None
            wis.append(wi)
        return wis

    @abstractmethod
    def process_batch(self, wis:List[object]): pass

    def run(self):
        super().thread_started()
        while self.get_state() == RTPWrkTRun:
            wis = self.next_batch()
            if wis is None:
                break
            self.process_batch(wis)

    def stop(self):
        self.inf_queue.put(None)
        super().stop()
