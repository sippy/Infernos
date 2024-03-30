from uuid import uuid4, UUID

import torch

class STTRequest():
    lang: str
    audio: torch.Tensor
    text_cb: callable
    def __init__(self, audio:torch.Tensor, text_cb:callable, lang:str):
        self.lang, self.audio, self.text_cb = lang, audio, text_cb

class STTResult():
    text: str
    no_speech_prob: float
    def __init__(self, text:str, no_speech_prob:float):
        self.text = text
        self.no_speech_prob = no_speech_prob

class STTSession():
    debug = False
    id: UUID
    lang: str = 'en'

    def __init__(self, stt):
        super().__init__()
        self.id = uuid4()
        self.stt = stt

    def stop(self):
        if self.debug: print('STTSession.stop')
        del self.stt

    def soundin(self, req:STTRequest):
        if self.debug: print(f'STTSession.soundin({len(req.audio)=})')
        self.stt.infer(req)
