#try: import intel_extension_for_pytorch as ipex
#except ModuleNotFoundError: ipex = None

from typing import Dict
from uuid import UUID

import ray

from Cluster.InfernSTTWorker import InfernSTTWorker
from Cluster.STTSession import STTSession, STTRequest

@ray.remote(num_gpus=0.25, resources={"stt": 1})
class InfernSTTActor():
    debug = False
    sessions: Dict[UUID, STTSession]
    stt: InfernSTTWorker

    def __init__(self):
        super().__init__()
        self.sessions = {}

    def start(self):
        for device in ('cuda', 'cpu'):
            try:
                self.stt = InfernSTTWorker(device)
            except (ValueError, RuntimeError):
                continue
            break
        else:
            raise RuntimeError('Failed to initialize STT')
        self.stt.start()

    def stop(self):
        self.stt.stop()

    def new_stt_session(self, keep_context:bool=False):
        if self.debug: print('InfernSTTActor.new_stt_session')
        sess = STTSession(self.stt, keep_context)
        self.sessions[sess.id] = sess
        return sess.id

    def stt_session_end(self, sess_id):
        if self.debug: print('InfernSTTActor.stt_session_end')
        sess = self.sessions[sess_id]
        sess.stop()
        del self.sessions[sess_id]

    def stt_session_soundin(self, sess_id, req:STTRequest):
        if self.debug: print('InfernSTTActor.stt_session_soundin')
        sess = self.sessions[sess_id]
        sess.soundin(req)
        return sess_id
