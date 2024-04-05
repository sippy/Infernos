try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from typing import Dict, Optional
from uuid import UUID

import ray

from Cluster.TTSSession import TTSSession2
from Cluster.InfernTTSWorker import InfernTTSWorker

@ray.remote(num_gpus=1, resources={"tts": 1})
class InfernTTSActor():
    sessions: Dict[UUID, TTSSession2]
    tts: InfernTTSWorker

    def __init__(self):
        super().__init__()
        self.sessions = {}

    def start(self, lang:str='en', output_sr:int=16000):
        self.tts = InfernTTSWorker(lang, output_sr)
        self.tts.start()

    def stop(self):
        self.tts.stop()

    def new_tts_session(self):
        tts_actr = ray.get_runtime_context().current_actor
        rgen = TTSSession2(self.tts, tts_actr)
        self.sessions[rgen.id] = rgen
        return rgen.id

    def tts_session_start(self, rgen_id, soundout:callable):
        rgen = self.sessions[rgen_id]
        rgen.start(soundout)

    def tts_session_say(self, rgen_id, text, done_cb:Optional[callable]=None, speaker_id=None):
        rgen = self.sessions[rgen_id]
        rgen.say(text, speaker_id, done_cb)

    def tts_session_end(self, rgen_id):
        rgen = self.sessions[rgen_id]
        rgen.stop()
        del self.sessions[rgen_id]
