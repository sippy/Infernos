from typing import List, Optional
from uuid import uuid4, UUID
from fractions import Fraction
from functools import partial
from threading import Lock

import torch

class STTRequest():
    lang: str
    audio: torch.Tensor
    text_cb: callable
    mode: str = 'transcribe'
    timestamps: bool = False
    def __init__(self, audio:torch.Tensor, text_cb:callable, lang:str):
        self.lang, self.audio, self.text_cb = lang, audio, text_cb

class STTResult():
    text: str
    no_speech_prob: float
    duration: Fraction
    offsets: Optional[List]=None
    def __init__(self, text:str, no_speech_prob:float, duration:Fraction):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.duration = duration

class STTSession():
    debug = False
    id: UUID
    lang: str = 'en'
    context: List[int]
    state_lock: Lock
    busy: bool = False
    pending: List[STTRequest]

    def __init__(self, stt, keep_context:bool):
        super().__init__()
        self.id = uuid4()
        self.stt = stt
        self.state_lock = Lock()
        self.context = [] if keep_context else None
        self.pending = []

    def stop(self):
        if self.debug: print('STTSession.stop')
        with self.state_lock:
            del self.stt, self.pending

    def soundin(self, req:STTRequest):
        if self.debug: print(f'STTSession.soundin({len(req.audio)=})')
        with self.state_lock:
            if self.busy:
                self.pending.append(req)
                return
            self.busy = True
        req.text_cb = partial(self.tts_out, req.text_cb)
        self.stt.infer((req, self.context))

    def tts_out(self, text_cb, result:STTResult):
        with self.state_lock:
            if not hasattr(self, 'stt'):
                return
            if self.debug: print(f'STTSession.tts_out({result.text=})')
            assert self.busy
            if self.pending:
                req = self.pending.pop(0)
                req.text_cb = partial(self.tts_out, req.text_cb)
                self.stt.infer((req, self.context))
            else:
                self.busy = False
        text_cb(result=result)
