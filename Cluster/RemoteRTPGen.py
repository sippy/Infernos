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

    def soundout(self, chunk):
        if not isinstance(chunk, TTSSMarkerGeneric):
            chunk = chunk.to('cpu')
        return ray.get(self.rtp_actr.rtp_session_soundout.remote(self.sess_id, chunk))

    def end(self):
        return ray.get(self.rtp_actr.rtp_session_end.remote(self.sess_id))

    def join(self):
        return ray.get(self.rtp_actr.rtp_session_join.remote(self.sess_id))

class RemoteRTPGenFromId(RemoteRTPGen):
    def __init__(self, rtp_actr, sess_id):
        self.rtp_actr = rtp_actr
        self.sess_id = sess_id
