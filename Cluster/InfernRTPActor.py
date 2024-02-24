try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from uuid import uuid4, UUID
from time import monotonic

from ray import ray

from sippy.Udp_server import Udp_server, Udp_server_opts
from sippy.misc import local4remote

from TTSRTPOutput import TTSRTPOutput
from SIP.InfernRTPIngest import InfernRTPIngest

class InfernRTPEPoint():
    id: UUID
    dl_file = None
    def __init__(self, rtp_target):
        self.id = uuid4()
        self.rtp_target = rtp_target
        self.ring = InfernRTPIngest()
        rtp_laddr = local4remote(rtp_target[0])
        rserv_opts = Udp_server_opts((rtp_laddr, 0), self.ring.rtp_received)
        rserv_opts.nworkers = 1
        self.rserv = Udp_server({}, rserv_opts)
        for dev in ('xpu' if ipex is not None else 'cuda', 'cpu'):
            try: self.writer = TTSRTPOutput(0, dev)
            except RuntimeError: continue
            break
        self.writer.set_pkt_send_f(self.send_pkt)
        if self.dl_file is not None:
            self.writer.enable_datalog(self.dl_file)
        self.ring.start()
        self.writer.start()

    def send_pkt(self, pkt):
        self.rserv.send_to(pkt, self.rtp_target)

@ray.remote(resources={"rtp": 1})
class InfernRTPActor():
    sessions: dict
    def __init__(self):
        self.sessions = {}

    def stdtss(self):
        return f'{monotonic():4.3f}'

    def new_rtp_session(self, rtp_target):
        print(f'{self.stdtss()}: new_rtp_session')
        rep = InfernRTPEPoint(rtp_target)
        self.sessions[rep.id] = rep
        return (rep.id, rep.rserv.uopts.laddress)

    def end_rtp_session(self, rtp_id):
        print(f'{self.stdtss()}: end_rtp_session')
        rep = self.sessions[rtp_id]
        rep.writer.end()

    def soundout_rtp_session(self, rtp_id, chunk):
        print(f'{self.stdtss()}: soundout_rtp_session')
        rep = self.sessions[rtp_id]
        return rep.writer.soundout(chunk)

    def join_rtp_session(self, rtp_id):
        print(f'{self.stdtss()}: join_rtp_session')
        rep = self.sessions[rtp_id]
        rep.writer.join()
        rep.ring.stop()
        rep.ring.join()
        rep.rserv.shutdown()
        del self.sessions[rtp_id]
