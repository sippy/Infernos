from typing import Optional
from uuid import UUID

import ray
from ray.exceptions import RayTaskError

from .TTSSession import TTSRequest

class TTSSessionError(Exception):
    pass

class RemoteTTSSession():
    sess_id: UUID
    def __init__(self, tts_actr):
        super().__init__()
        self.tts_actr = tts_actr
        try: self.sess_id = ray.get(tts_actr.new_tts_session.remote())
        except RayTaskError as e: raise TTSSessionError("new_tts_session() failed") from e

    def start(self, soundout:callable):
        return ray.get(self.tts_actr.tts_session_start.remote(self.sess_id, soundout))

    def end(self):
        return ray.get(self.tts_actr.tts_session_end.remote(self.sess_id))

    def say(self, req:TTSRequest):
        return self.tts_actr.tts_session_say.remote(rgen_id=self.sess_id, req=req)

    def stop_saying(self, rsay_id:UUID):
        return self.tts_actr.tts_session_stop_saying.remote(rgen_id=self.sess_id, rsay_id=rsay_id)
