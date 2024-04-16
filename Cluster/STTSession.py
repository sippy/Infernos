from typing import List, Optional
from uuid import uuid4, UUID
from fractions import Fraction
from functools import partial
from threading import Lock
from time import monotonic

import torch

from Core.AudioChunk import AudioChunk

class STTRequest():
    lang: str
    chunk: AudioChunk
    text_cb: callable
    mode: str = 'transcribe'
    timestamps: bool = False
    stime:float
    def __init__(self, chunk:AudioChunk, text_cb:callable, lang:str):
        self.stime = monotonic()
        self.lang, self.chunk, self.text_cb = lang, chunk, text_cb

class STTResult():
    text: str
    no_speech_prob: float
    duration: Fraction
    offsets: Optional[List]=None
    inf_time: float
    def __init__(self, text:str, no_speech_prob:float, req:STTRequest):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.duration = Fraction(len(req.chunk.audio), req.chunk.samplerate)
        self.inf_time = monotonic() - req.stime

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
        if self.debug: print(f'STTSession.soundin({len(req.chunk.audio)=})')
        if req.chunk.samplerate != self.stt.sample_rate:
            req.chunk.resample(self.stt.sample_rate)
        req.chunk.audio = req.chunk.audio.numpy()
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
