from typing import Optional
from functools import partial

import ray
from ray.exceptions import RayTaskError

class RTPGenError(Exception):
    pass

class RemoteRTPGen():
    def __init__(self, rtp_actr, target):
        self.rtp_actr = rtp_actr
        fut = rtp_actr.new_rtp_session.remote(target)
        try: self.sess_id, self.rtp_address = ray.get(fut)
        except RayTaskError as e: raise RTPGenError("new_rtp_session() failed") from e

    def connect(self, vad_chunk_in:callable, audio_in:Optional[callable]=None):
        return self.rtp_actr.rtp_session_connect.remote(self.sess_id, vad_chunk_in, audio_in)

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
