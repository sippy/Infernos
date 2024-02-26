import ray
from ray.exceptions import RayTaskError

from TTSRTPOutput import TTSSMarkerGeneric

class RTPGenError(Exception):
    pass

class RemoteRTPGen():
    def __init__(self, rtp_actr, target):
        self.rtp_actr = rtp_actr
        try: self.sess_id, self.rtp_address = ray.get(rtp_actr.new_rtp_session.remote(target))
        except RayTaskError as e: raise RTPGenError("new_rtp_session() failed") from e

    def soundout(self, chunk):
        if not isinstance(chunk, TTSSMarkerGeneric):
            chunk = chunk.to('cpu')
        return ray.get(self.rtp_actr.soundout_rtp_session.remote(self.sess_id, chunk))

    def end(self):
        return ray.get(self.rtp_actr.end_rtp_session.remote(self.sess_id))

    def join(self):
        return ray.get(self.rtp_actr.join_rtp_session.remote(self.sess_id))