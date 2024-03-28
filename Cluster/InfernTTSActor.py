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

    def __init__(self, lang:str='en'):
        super().__init__()
        self.sessions = {}
        self.tts = InfernTTSWorker(lang)

    def new_tts_session(self):
        rgen = TTSSession(self.tts)
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
