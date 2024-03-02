try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from typing import Dict
from uuid import UUID

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

    def new_tts_session(self, sip_sess_id):
        rgen = TTSSession(self.tts, lambda: self.sip_actr.sess_term.remote(sip_sess_id))
        self.sessions[rgen.id] = rgen
        return rgen.id

    def start_tts_session(self, rgen_id, rtp_sess_id, text):
        rgen = self.sessions[rgen_id]
        rtp_address = rgen.start(self.tts_actr, self.rtp_actr, rtp_sess_id, text)
        return rtp_address

    def tts_session_next_sentence(self, rgen_id):
        rgen = self.sessions[rgen_id]
        rgen.next_sentence()

    def tts_session_stopintro(self, rgen_id):
        rgen = self.sessions[rgen_id]
        rgen.stopintro()

    def tts_session_say(self, rgen_id, text):
        rgen = self.sessions[rgen_id]
        rgen.say(text)

    def end_tts_session(self, rgen_id):
        rgen = self.sessions[rgen_id]
        rgen.stop()
        del self.sessions[rgen_id]
