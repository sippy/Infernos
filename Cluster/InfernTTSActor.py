try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

import ray

from SIP.InfernRTPGen import InfernRTPGen

from TTS import TTS

@ray.remote
class InfernTTSActor():
    sessions: dict
    tts: TTS

    def __init__(self, rtp_actr):
        super().__init__()
        self.sessions = {}
        self.tts = TTS()
        self.rtp_actr = rtp_actr

    def new_tts_session(self):
        rgen = InfernRTPGen(self.tts, self.sess_term)
        self.sessions[rgen.id] = rgen
        return rgen.id

    def start_tts_session(self, rgen_id, text, target):
        rgen = self.sessions[rgen_id]
        rtp_address = rgen.start(self.rtp_actr, text, target)
        return rtp_address

    def end_tts_session(self, rgen_id):
        rgen = self.sessions[rgen_id]
        rgen.stop()
        del self.sessions[rgen_id]

    def sess_term(self):
        print('sess_term')
