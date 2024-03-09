from uuid import uuid4, UUID

class STTSession():
    debug = True
    id: UUID

    def __init__(self, stt):
        super().__init__()
        self.id = uuid4()

    def stop(self):
        if self.debug: print('STTSession.stop')

    def soundin(self, chunk):
        if self.debug: print(f'STTSession.soundin({len(chunk)=})')