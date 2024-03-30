from uuid import uuid4, UUID

import torch

class STTRequest():
    lang: str
    audio: torch.Tensor
    text_cb: callable
    def __init__(self, audio:torch.Tensor, text_cb:callable, lang:str):
        self.lang, self.audio, self.text_cb = lang, audio, text_cb

class STTSession():
    debug = False
    id: UUID
    lang: str = 'en'

    def __init__(self, stt, text_cb):
        super().__init__()
        self.id = uuid4()
        self.stt = stt
        self.text_cb = text_cb

    def stop(self):
        if self.debug: print('STTSession.stop')
        del self.stt

    def soundin(self, req:STTRequest):
        if self.debug: print(f'STTSession.soundin({len(req.audio)=})')
        self.stt.infer(req)
