from typing import List, Optional, Union
from uuid import uuid4, UUID
from fractions import Fraction
from functools import partial
from threading import Lock
from time import monotonic

from Core.AudioChunk import AudioChunk, VadAudioChunk

class STTRequest():
    lang: str
    chunk: AudioChunk
    text_cb: callable
    mode: str = 'transcribe'
    timestamps: bool = False
    stime: float
    max_ns_prob: float = 0.5
    def __init__(self, chunk:AudioChunk, text_cb:callable, lang:str):
        self.stime = monotonic()
        self.lang, self.chunk, self.text_cb = lang, chunk, text_cb

class STTSentinel():
    stime: float
    text_cb: callable
    def __init__(self, signal:str, text_cb:callable):
        self.stime = monotonic()
        self.signal, self.text_cb = signal, text_cb

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

    def soundin(self, req:Union[STTRequest,STTSentinel]):
        if self.debug:
            if isinstance(req, STTRequest):
                print(f'STTSession.soundin({len(req.chunk.audio)=})')
            else:
                print(f'STTSession.soundin({req=})')
        results = []
        with self.state_lock:
            self.pending.append(req)
            if self.busy:
                return
            assert len(self.pending) == 1
            self.busy = True
            self._process_pending_stt_lckd(results)
        for cb, r in results:
            cb(result=r)

    def _process_pending_stt_lckd(self, results:List):
        while self.pending:
            req = self.pending.pop(0)
            if isinstance(req, STTRequest):
                if isinstance(req.chunk, VadAudioChunk):
                    nr = next((r for r in self.pending if isinstance(r, STTRequest)), None)
                    if nr is not None and isinstance(nr.chunk, VadAudioChunk):
                        ca, cb = req.chunk, nr.chunk
                        if cb.tpos() + cb.duration() - ca.tpos() < self.stt.max_chunk_duration:
                            ca.append(cb)
                            self.pending.remove(nr)
                            self.pending.insert(0, req)
                            continue
                if req.chunk.samplerate != self.stt.sample_rate:
                    req.chunk.resample(self.stt.sample_rate)
                req.chunk.audio = req.chunk.audio.numpy()
                text_cb = partial(self.stt_out, req.text_cb)
                self.stt.infer((req, text_cb, self.context))
                break
            if all(isinstance(r, STTRequest) for r in self.pending):
                results.append((req.text_cb, req))
        else:
            self.busy = False

    def stt_out(self, text_cb, result:STTResult):
        results = [(text_cb, result)]
        with self.state_lock:
            if not hasattr(self, 'stt'):
                return
            if self.debug: print(f'STTSession.stt_out({result.text=})')
            assert self.busy
            self._process_pending_stt_lckd(results)
        for cb, r in results:
            cb(result=r)
