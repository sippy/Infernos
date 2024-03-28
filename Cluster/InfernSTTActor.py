#try: import intel_extension_for_pytorch as ipex
#except ModuleNotFoundError: ipex = None

import ray

from Cluster.InfernSTTWorker import InfernSTTWorker
from Cluster.STTSession import STTSession

@ray.remote(num_gpus=1, resources={"stt": 1})
class InfernSTTActor():
    debug = True
    sessions: dict
    stt: InfernSTTWorker

    def __init__(self):
        super().__init__()
        self.sessions = {}

    def start(self):
        self.stt = InfernSTTWorker('cuda')
        self.stt.start()

    def stop(self):
        self.stt.stop()

    def new_stt_session(self, activate_cb):
        if self.debug: print('InfernSTTActor.new_stt_session')
        sess = STTSession(self.stt, activate_cb)
        self.sessions[sess.id] = sess
        return sess.id

    def stt_session_end(self, sess_id):
        if self.debug: print('InfernSTTActor.stt_session_end')
        sess = self.sessions[sess_id]
        sess.stop()
        del self.sessions[sess_id]

    def stt_session_soundin(self, sess_id, chunk):
        if self.debug: print('InfernSTTActor.stt_session_soundin')
        sess = self.sessions[sess_id]
        sess.soundin(chunk)
        return sess_id
