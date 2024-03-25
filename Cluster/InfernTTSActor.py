try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from typing import Dict, Optional
from uuid import UUID
from functools import partial

import ray

from Cluster.TTSSession import TTSSession
from Cluster.InfernTTSWorker import InfernTTSWorker

@ray.remote
class InfernTTSActor():
    sessions: Dict[UUID, TTSSession]
    tts: InfernTTSWorker

    def __init__(self, rtp_actr, sip_actr):
        super().__init__()
        self.sessions = {}
        self.tts = InfernTTSWorker()
        self.rtp_actr = rtp_actr
        self.sip_actr = sip_actr
        self.tts_actr = ray.get_runtime_context().current_actor

    def new_tts_session(self):
        rgen = TTSSession(self.tts)
        self.sessions[rgen.id] = rgen
        return rgen.id

    def start_tts_session(self, rgen_id, rtp_sess_id):
        rgen = self.sessions[rgen_id]
        rtp_address = rgen.start(self.rtp_actr, rtp_sess_id)
        return rtp_address

    def tts_session_say(self, rgen_id, text, done_cb:Optional[ray.ObjectRef]=None):
        rgen = self.sessions[rgen_id]
        rgen.say(text, done_cb)

    def end_tts_session(self, rgen_id):
        rgen = self.sessions[rgen_id]
        rgen.stop()
        del self.sessions[rgen_id]
