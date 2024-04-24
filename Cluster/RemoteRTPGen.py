from typing import Optional
from functools import partial

import ray
from ray.exceptions import RayTaskError

from Cluster.InfernRTPActor import RTPSessNotFoundErr
from RTP.AudioInput import AudioInput
from RTP.RTPParams import RTPParams

class RTPGenError(Exception):
    pass

class RemoteRTPGen():
    def __init__(self, rtp_actr, params:RTPParams):
        self.rtp_actr = rtp_actr
        fut = rtp_actr.new_rtp_session.remote(params)
        try: self.sess_id, self.rtp_address = ray.get(fut)
        except RayTaskError as e: raise RTPGenError("new_rtp_session() failed") from e

    def connect(self, ain:AudioInput):
        return self.rtp_actr.rtp_session_connect.remote(self.sess_id, ain)

    def update(self, params:RTPParams):
        return ray.get(self.rtp_actr.rtp_session_update.remote(self.sess_id, params))

    def get_soundout(self) -> callable:
        return partial(self.rtp_actr.rtp_session_soundout.remote, rtp_id=self.sess_id)

    def soundout(self, chunk):
        self.rtp_actr.rtp_session_soundout.remote(rtp_id=self.sess_id, chunk=chunk)

    def end(self, relaxed:bool=True):
        return self.rtp_actr.rtp_session_end.remote(self.sess_id, relaxed)

    def join(self):
        try: ray.get(self.rtp_actr.rtp_session_join.remote(self.sess_id))
        except RTPSessNotFoundErr: pass
