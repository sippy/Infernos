from typing import Optional
from queue import Queue

import torch
import ctranslate2
import transformers


from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun
from Cluster.STTSession import STTSession

class STTWI():
    stt_sess: STTSession
    audio: torch.Tensor
    res_queue: Queue

class InfernSTTWorker(InfernWrkThread):
    model: ctranslate2.models.Whisper
    processor: transformers.WhisperProcessor
    device: str
    inf_queue: Queue[Optional[STTWI]]
    def __init__(self, device: str):
        super().__init__()
        self.model = ctranslate2.models.Whisper("whisper-large-v3.ct2", device=device, compute_type="int8")
        self.processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.processor.eval()
        #self.model.eval()
        self.inf_queue = Queue()
        self.device = device

    def infer(self, stt_sess, audio, res_queue):
        wi = STTWI()
        wi.stt_sess, wi.audio, wi.res_queue = stt_sess, audio, res_queue
        self.inf_queue.put(wi)

    def run(self):
        super().thread_started()
        while self.get_state() == RTPWrkTRun:
            wi = self.inf_queue.get()
            if wi is None:
                break
            print(f'InfernSTTWorker.run: got {wi} from inf_queue')
            #with torch.no_grad():
            #    audio = wi.audio.to(self.device)
            #    res = wi.stt_sess.model(audio)
            #    wi.res_queue.put(res)

    def stop(self):
        self.inf_queue.put(None)
        super().stop()

