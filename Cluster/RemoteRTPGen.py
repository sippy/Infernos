from functools import partial

import ray
from ray.exceptions import RayTaskError

from TTSRTPOutput import TTSSMarkerGeneric

class RTPGenError(Exception):
    pass

class RemoteRTPGen():
    def __init__(self, rtp_actr, stt_sess_id, target):
        self.rtp_actr = rtp_actr
        try: self.sess_id, self.rtp_address = ray.get(rtp_actr.new_rtp_session.remote(target, stt_sess_id))
        except RayTaskError as e: raise RTPGenError("new_rtp_session() failed") from e

    def update(self, target):
        return ray.get(self.rtp_actr.rtp_session_update.remote(self.sess_id, target))

    def get_soundout(self) -> callable:
        return partial(self.rtp_actr.rtp_session_soundout.remote, rtp_id=self.sess_id)

    def end(self):
        return ray.get(self.rtp_actr.rtp_session_end.remote(self.sess_id))

    def join(self):
        return ray.get(self.rtp_actr.rtp_session_join.remote(self.sess_id))
