from uuid import uuid4, UUID

class STTSession():
    debug = True
    id: UUID
    lang: str = 'es'

    def __init__(self, stt, tts_sess_id):
        super().__init__()
        self.id = uuid4()
        self.stt = stt
        self.tts_sess_id = tts_sess_id

    def stop(self):
        if self.debug: print('STTSession.stop')
        del self.stt

    def soundin(self, chunk):
        if self.debug: print(f'STTSession.soundin({len(chunk)=})')
        self.stt.infer(self, chunk, self.tts_sess_id)
