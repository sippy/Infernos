try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from typing import Any, Tuple, Dict
from uuid import uuid4, UUID
from time import monotonic
from threading import Lock
from _thread import get_ident

from ray import ray

from sippy.Udp_server import Udp_server, Udp_server_opts
from sippy.misc import local4remote

from RTP.RTPOutputWorker import RTPOutputWorker,  TTSSMarkerGeneric, TTSSMarkerNewSent
from SIP.InfernRTPIngest import InfernRTPIngest, RTPInStream


class InfernRTPEPoint():
    debug = True
    #devs = ('xpu' if ipex is not None else 'cuda', 'cpu')
    devs = ('cpu',)
    id: UUID
    dl_file = None
    firstframe = True
    rtp_target: Tuple[str, int]
    rtp_target_lock: Lock
    def __init__(self, rtp_target:Tuple[str, int], vad_chunk_in:callable, ring):
        self.id = uuid4()
        assert isinstance(rtp_target, tuple) and len(rtp_target) == 2
        self.rtp_target = rtp_target
        self.rtp_target_lock = Lock()
        for dev in self.devs:
            try:
                self.writer = RTPOutputWorker(0, dev)
                self.rsess = RTPInStream(ring, vad_chunk_in, dev)
            except RuntimeError:
                if dev == self.devs[-1]: raise
            else: break
        rtp_laddr = local4remote(rtp_target[0])
        rserv_opts = Udp_server_opts((rtp_laddr, 0), self.rtp_received)
        rserv_opts.nworkers = 1
        rserv_opts.direct_dispatch = True
        self.rserv = Udp_server({}, rserv_opts)
        self.writer.set_pkt_send_f(self.send_pkt)
        if self.dl_file is not None:
            self.writer.enable_datalog(self.dl_file)
        self.writer.start()

    def send_pkt(self, pkt):
        with self.rtp_target_lock:
            rtp_target = self.rtp_target
        self.rserv.send_to(pkt, rtp_target)

    def rtp_received(self, data, address, udp_server, rtime):
        #self.dprint(f"InfernRTPIngest.rtp_received: len(data) = {len(data)}")
        with self.rtp_target_lock:
            if address != self.rtp_target:
                self.dprint(f"InfernRTPIngest.rtp_received: address mismatch {address=} {self.rtp_target=}")
                return
        self.rsess.rtp_received(data, address, rtime)

    def update(self, rtp_target:Tuple[str, int]):
        assert isinstance(rtp_target, tuple) and len(rtp_target) == 2
        with self.rtp_target_lock:
            self.rtp_target = rtp_target
        self.rsess.stream_update()

    def shutdown(self):
        self.writer.join()
        self.rserv.shutdown()
        self.rserv, self.writer = (None, None)

    def __del__(self):
        if self.debug:
            print('InfernRTPEPoint.__del__')

    def soundout(self, chunk, stdtss):
        ismark = isinstance(chunk, TTSSMarkerGeneric)
        if self.firstframe or ismark:
            print(f'{stdtss()}: rtp_session_soundout: {"mark" if ismark else "data"}')
            self.firstframe = False
        if ismark and isinstance(chunk, TTSSMarkerNewSent):
            self.firstframe = True
        if not ismark:
            chunk = chunk.to(self.writer.device)
        return self.writer.soundout(chunk)

@ray.remote(resources={"rtp": 1})
class InfernRTPActor():
    device = 'cpu'
    sessions: Dict[UUID, InfernRTPEPoint]
    ring: InfernRTPIngest
    def __init__(self):
        self.sessions = {}

    def stdtss(self):
        return f'{monotonic():4.3f}'

    def new_rtp_session(self, rtp_target, vad_chunk_in:callable):
        print(f'{self.stdtss()}: new_rtp_session')
        rep = InfernRTPEPoint(rtp_target, vad_chunk_in, self.ring)
        self.sessions[rep.id] = rep
        return (rep.id, rep.rserv.uopts.laddress)

    def rtp_session_end(self, rtp_id):
        print(f'{self.stdtss()}: rtp_session_end')
        rep = self.sessions[rtp_id]
        rep.writer.end()

    def rtp_session_soundout(self, rtp_id, chunk):
        rep = self.sessions[rtp_id]
        return rep.soundout(chunk, self.stdtss)

    def rtp_session_join(self, rtp_id):
        print(f'{self.stdtss()}: rtp_session_join')
        rep = self.sessions[rtp_id]
        rep.shutdown()
        del self.sessions[rtp_id]

    def rtp_session_update(self, rtp_id, rtp_target):
        print(f'{self.stdtss()}: rtp_session_update')
        rep = self.sessions[rtp_id]
        rep.update(rtp_target)

    def start(self):
        self.ring = InfernRTPIngest()
        self.ring.start()

    def loop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        rval = ED2.loop()
        self.ring.stop()
        self.ring.join()
        return rval

    def stop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.callFromThread(ED2.breakLoop, 0)
