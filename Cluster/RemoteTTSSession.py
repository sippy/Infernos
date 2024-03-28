from typing import Optional

import ray
from ray.exceptions import RayTaskError

class TTSSessionError(Exception):
    pass

class RemoteTTSSession():
    def __init__(self, tts_actr):
        super().__init__()
        self.tts_actr = tts_actr
        try: self.sess_id = ray.get(tts_actr.new_tts_session.remote())
        except RayTaskError as e: raise TTSSessionError("new_tts_session() failed") from e

    def start(self, soundout:callable):
        return ray.get(self.tts_actr.start_tts_session.remote(self.sess_id, soundout))

    def end(self):
        return ray.get(self.tts_actr.end_tts_session.remote(self.sess_id))

    def say(self, text, done_cb:Optional[callable]=None):
        return self.tts_actr.tts_session_say.remote(self.sess_id, text, done_cb)
