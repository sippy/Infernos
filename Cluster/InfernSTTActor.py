#try: import intel_extension_for_pytorch as ipex
#except ModuleNotFoundError: ipex = None

import ray

from Cluster.InfernSTTWorker import InfernSTTWorker
from Cluster.STTSession import STTSession

#from STT import STT

@ray.remote(num_gpus=1, resources={"stt": 1})
class InfernSTTActor():
    debug = True
    sessions: dict
    stt: InfernSTTWorker
    tts_actr: ray.actor

    def __init__(self):
        super().__init__()
        self.sessions = {}
        #self.stt = STT()

    def start(self, tts_actr):
        self.stt = InfernSTTWorker(tts_actr, 'cuda')
        self.stt.start()

    def stop(self):
        self.stt.stop()

    def new_stt_session(self, tts_sess_id):
        if self.debug: print('InfernSTTActor.new_stt_session')
        sess = STTSession(self.stt, tts_sess_id)
        self.sessions[sess.id] = sess
        return sess.id

    def end_stt_session(self, sess_id):
        if self.debug: print('InfernSTTActor.end_stt_session')
        sess = self.sessions[sess_id]
        sess.stop()
        del self.sessions[sess_id]

    def stt_session_soundin(self, sess_id, chunk):
        if self.debug: print('InfernSTTActor.stt_session_soundin')
        sess = self.sessions[sess_id]
        sess.soundin(chunk)
        return sess_id
