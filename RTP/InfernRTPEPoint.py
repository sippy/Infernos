from typing import Tuple
from uuid import uuid4, UUID
from threading import Lock

from sippy.Udp_server import Udp_server, Udp_server_opts
from sippy.misc import local4remote

from RTP.RTPOutputWorker import RTPOutputWorker,  TTSSMarkerGeneric, TTSSMarkerNewSent
from SIP.InfernRTPIngest import RTPInStream

class InfernRTPEPoint():
    debug = False
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
                self.writer = RTPOutputWorker(dev)
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
