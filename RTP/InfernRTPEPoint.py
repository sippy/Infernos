from typing import Tuple, Union
from uuid import uuid4, UUID
from threading import Lock

from sippy.Udp_server import Udp_server, Udp_server_opts
from sippy.misc import local4remote

from config.InfernGlobals import InfernGlobals as IG
from Core.AudioChunk import AudioChunk
from Core.AStreamMarkers import ASMarkerGeneric, ASMarkerNewSent
from RTP.RTPOutputWorker import RTPOutputWorker
from RTP.InfernRTPIngest import RTPInStream
from RTP.AudioInput import AudioInput
from RTP.RTPParams import RTPParams
from RTP.InfernRTPIngest import InfernRTPIngest
from RTP.InfernRTPConf import InfernRTPConf

class InfernRTPEPoint():
    debug: bool = False
    id: UUID
    dl_file = None
    firstframe = True
    rtp_params:RTPParams
    state_lock: Lock
    def __init__(self, rc:InfernRTPConf, rtp_params:RTPParams, ring:InfernRTPIngest, get_direct_soundout:callable):
        self.id = uuid4()
        self.rtp_params = rtp_params
        self.state_lock = Lock()
        self.writer = RTPOutputWorker('cpu', rtp_params)
        self.rsess = RTPInStream(ring, rtp_params, get_direct_soundout)
        rtp_laddr = local4remote(rtp_params.rtp_target[0])
        rserv_opts = Udp_server_opts((rtp_laddr, rc.palloc), self.rtp_received)
        rserv_opts.nworkers = 1
        rserv_opts.direct_dispatch = True
        self.rserv = Udp_server({}, rserv_opts)
        self.writer_setup()

    def writer_setup(self):
        self.writer.set_pkt_send_f(self.send_pkt)
        if self.dl_file is not None:
            self.writer.enable_datalog(self.dl_file)
        self.writer.start()

    def send_pkt(self, pkt):
        with self.state_lock:
            rtp_target = self.rtp_params.rtp_target
        self.rserv.send_to(pkt, rtp_target)

    def rtp_received(self, data, address, udp_server, rtime):
        #self.dprint(f"InfernRTPIngest.rtp_received: len(data) = {len(data)}")
        with self.state_lock:
            if address != self.rtp_params.rtp_target:
                if self.debug:
                    print(f"InfernRTPIngest.rtp_received: address mismatch {address=} {self.rtp_params.rtp_target=}")
                return
        self.rsess.rtp_received(data, address, rtime)

    def update(self, rtp_params:RTPParams):
        with self.state_lock:
            self.rtp_params.rtp_target = rtp_params.rtp_target
            if self.rtp_params.out_ptime != rtp_params.out_ptime:
                self.writer.end()
                self.writer.join()
                self.writer = RTPOutputWorker('cpu', rtp_params)
                self.writer_setup()
        self.rsess.stream_update()

    def connect(self, ain:AudioInput):
        self.rsess.stream_connect(ain)

    def shutdown(self):
        with self.state_lock:
            self.writer.join()
            self.rserv.shutdown()
            self.rserv, self.writer = (None, None)

    def __del__(self):
        if self.debug:
            print('InfernRTPEPoint.__del__')

    def soundout(self, chunk:Union[AudioChunk, ASMarkerGeneric]):
        ismark = isinstance(chunk, ASMarkerGeneric)
        if self.firstframe or ismark:
            print(f'{IG.stdtss()}: rtp_session_soundout[{str(self.id)[:6]}]: {"mark" if ismark else chunk.audio.size(0)}')
            self.firstframe = False
        if ismark and isinstance(chunk, ASMarkerNewSent):
            self.firstframe = True
        with self.state_lock:
            if self.writer is None: return
            return self.writer.soundout(chunk)
