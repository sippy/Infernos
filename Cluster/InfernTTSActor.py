#try: import intel_extension_for_pytorch as ipex
#except ModuleNotFoundError: ipex = None

from typing import Dict, Optional
from uuid import UUID

import ray

from Cluster.TTSSession import TTSSession, TTSRequest
from Cluster.InfernTTSWorker import InfernTTSWorker

@ray.remote(num_gpus=0.25, resources={"tts": 1})
class InfernTTSActor():
    sessions: Dict[UUID, TTSSession]
    tts: InfernTTSWorker

    def __init__(self):
        super().__init__()
        self.sessions = {}

    def start(self, lang:str='en', output_sr:int=16000, device=None):
        self.tts = InfernTTSWorker(lang, output_sr, device)
        self.tts.start()

    def stop(self):
        self.tts.stop()

    def get_rand_voice_id(self) -> int:
        return self.tts.get_rand_voice_id()

    def new_tts_session(self):
        tts_actr = ray.get_runtime_context().current_actor
        rgen = TTSSession(self.tts, tts_actr)
        self.sessions[rgen.id] = rgen
        return rgen.id

    def tts_session_start(self, rgen_id, soundout:callable):
        rgen = self.sessions[rgen_id]
        rgen.start(soundout)

    def tts_session_say(self, rgen_id, req:TTSRequest):
        rgen = self.sessions[rgen_id]
        rgen.say(req)

    def tts_session_end(self, rgen_id):
        rgen = self.sessions[rgen_id]
        rgen.stop()
        del self.sessions[rgen_id]
