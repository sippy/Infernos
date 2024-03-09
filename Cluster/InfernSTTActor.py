try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

import ray

from Cluster.STTSession import STTSession

#from STT import STT

@ray.remote
class InfernSTTActor():
    debug = True
    sessions: dict
#    stt: STT

    def __init__(self):
        super().__init__()
        self.sessions = {}
        #self.stt = STT()
        self.stt = None

    def new_stt_session(self):
        if self.debug: print('InfernSTTActor.new_stt_session')
        sess = STTSession(self.stt)
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
