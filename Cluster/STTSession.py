from uuid import uuid4, UUID

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

    def soundin(self, chunk):
        if self.debug: print(f'STTSession.soundin({len(chunk)=})')
        self.stt.infer(self, chunk, self.text_cb)
