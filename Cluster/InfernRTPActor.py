try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from uuid import uuid4, UUID
from time import monotonic
from _thread import get_ident

from ray import ray

from sippy.Udp_server import Udp_server, Udp_server_opts
from sippy.misc import local4remote

from TTSRTPOutput import TTSRTPOutput,  TTSSMarkerGeneric
from SIP.InfernRTPIngest import InfernRTPIngest

class InfernRTPEPoint():
    debug = True
    devs = ('xpu' if ipex is not None else 'cuda', 'cpu')
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
        for dev in self.devs:
            try: self.writer = TTSRTPOutput(0, dev)
            except RuntimeError:
                if dev == self.devs[-1]: raise
            else: break
        self.writer.set_pkt_send_f(self.send_pkt)
        if self.dl_file is not None:
            self.writer.enable_datalog(self.dl_file)
        self.ring.start()
        self.writer.start()

    def send_pkt(self, pkt):
        self.rserv.send_to(pkt, self.rtp_target)

    def shutdown(self):
        self.writer.join()
        self.ring.stop()
        self.ring.join()
        self.rserv.shutdown()
        self.ring, self.rserv, self.writer = (None, None, None)

    def __del__(self):
        if self.debug:
            print('InfernRTPEPoint.__del__')

@ray.remote(resources={"rtp": 1})
class InfernRTPActor():
    sessions: dict
    firstframe = True
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
        ismark = isinstance(chunk, TTSSMarkerGeneric)
        if self.firstframe or ismark:
            print(f'{self.stdtss()}: soundout_rtp_session')
            self.firstframe = False
        rep = self.sessions[rtp_id]
        if not ismark:
            chunk = chunk.to(rep.writer.device)
        return rep.writer.soundout(chunk)

    def join_rtp_session(self, rtp_id):
        print(f'{self.stdtss()}: join_rtp_session')
        rep = self.sessions[rtp_id]
        rep.shutdown()
        del self.sessions[rtp_id]

    def loop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        return (ED2.loop())

    def stop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.callFromThread(ED2.breakLoop, 0)
