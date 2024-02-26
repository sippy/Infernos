import ray
from ray.exceptions import RayTaskError

class TTSSessionError(Exception):
    pass

class RemoteTTSSession():
    def __init__(self, tts_actr, sip_sess_id):
        super().__init__()
        self.tts_actr = tts_actr
        try: self.sess_id = ray.get(tts_actr.new_tts_session.remote(sip_sess_id))
        except RayTaskError as e: raise TTSSessionError("new_tts_session() failed") from e

    def start(self, text, target):
        return ray.get(self.tts_actr.start_tts_session.remote(self.sess_id, text, target))

    def end(self):
        return ray.get(self.tts_actr.end_tts_session.remote(self.sess_id))