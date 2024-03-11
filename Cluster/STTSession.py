from uuid import uuid4, UUID

class STTSession():
    debug = True
    id: UUID

    def __init__(self, stt):
        super().__init__()
        self.id = uuid4()
        self.stt = stt

    def stop(self):
        if self.debug: print('STTSession.stop')
        del self.stt

    def soundin(self, chunk):
        if self.debug: print(f'STTSession.soundin({len(chunk)=})')
        self.stt.infer(self, chunk, None)