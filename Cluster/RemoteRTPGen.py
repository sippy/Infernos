from functools import partial

import ray
from ray.exceptions import RayTaskError

class RTPGenError(Exception):
    pass

class RemoteRTPGen():
    def __init__(self, rtp_actr, vad_chunk_in:callable, target):
        self.rtp_actr = rtp_actr
        try: self.sess_id, self.rtp_address = ray.get(rtp_actr.new_rtp_session.remote(target, vad_chunk_in))
        except RayTaskError as e: raise RTPGenError("new_rtp_session() failed") from e

    def update(self, target):
        return ray.get(self.rtp_actr.rtp_session_update.remote(self.sess_id, target))

    def get_soundout(self) -> callable:
        return partial(self.rtp_actr.rtp_session_soundout.remote, rtp_id=self.sess_id)

    def soundout(self, chunk):
        self.rtp_actr.rtp_session_soundout.remote(rtp_id=self.sess_id, chunk=chunk)

    def end(self):
        return self.rtp_actr.rtp_session_end.remote(self.sess_id)

    def join(self):
        return ray.get(self.rtp_actr.rtp_session_join.remote(self.sess_id))
