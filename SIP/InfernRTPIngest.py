from typing import Optional, Tuple, Union
from queue import Queue

from rtpsynth.RtpJBuf import RtpJBuf, RTPFrameType, RTPParseError

from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun
from Core.VAD.ZlibVAD import ZlibVAD
from Core.Codecs.G711 import G711Codec


class WIPkt():
    def __init__(self, stream: 'RTPInStream', data, address, rtime):
        self.stream = stream
        self.data = data
        self.address = address
        self.rtime = rtime

class WIStreamUpdate():
    def __init__(self, stream: 'RTPInStream'):
        self.stream = stream

class RTPInStream():
    jb_size: int = 8
    input_sr: int = 8000
    last_output_lseq: Optional[int] = None
    vad: ZlibVAD
    codec: G711Codec
    output_sr: int = 16000
    npkts: int = 0
    def __init__(self, ring:'InfernRTPIngest', chunk_in, device:str):
        self.jbuf = RtpJBuf(self.jb_size)
        self.vad = ZlibVAD(self.input_sr)
        self.codec = G711Codec(self.output_sr).to(device)
        self.ring = ring
        self.chunk_in = chunk_in

    def rtp_received(self, data, address, rtime):
        #self.dprint(f"InfernRTPIngest.rtp_received: len(data) = {len(data)}")
        self.ring.pkt_queue.put(WIPkt(self, data, address, rtime))

    def stream_update(self):
        self.ring.pkt_queue.put(WIStreamUpdate(self))

    def _proc_in_tread(self, wi:Union[WIPkt,WIStreamUpdate]):
        if isinstance(wi, WIStreamUpdate):
            print("InfernRTPIngest.run: stream update")
            self.jbuf = RtpJBuf(self.jb_size)
            self.last_output_lseq = None
            return
        data, address, rtime = wi.data, wi.address, wi.rtime
        try:
            res = self.jbuf.udp_in(data)
        except RTPParseError as e:
            self.ring.dprint(f"InfernRTPIngest.run: RTPParseError: {e}")
            return
        self.npkts += 1
        if self.npkts == 1:
            self.ring.dprint(f"InfernRTPIngest.run: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
        for pkt in res:
            if pkt.content.type == RTPFrameType.ERS: print(f"InfernRTPIngest.run: ERS packet received {pkt.content.lseq_start=}, {pkt.content.lseq_end=}")
            assert pkt.content.type != RTPFrameType.ERS
            if self.npkts < 10:
                self.ring.dprint(f"InfernRTPIngest.run: {pkt.content.frame.rtp.lseq=}")
            assert self.last_output_lseq is None or pkt.content.frame.rtp.lseq == self.last_output_lseq + 1
            self.last_output_lseq = pkt.content.frame.rtp.lseq
            if self.npkts < 10:
                self.ring.dprint(f"InfernRTPIngest.run: {len(pkt.rtp_data)=}, {type(pkt.rtp_data)=}")
            out = self.vad.ingest(pkt.rtp_data)
            if out is None: continue
            chunk = self.codec.decode(out.chunk, resample=True)
            self.ring.dprint(f"InfernRTPIngest.run: active chunk: {len(chunk)=}")
            self.chunk_in(chunk.numpy())
        if self.npkts < 10 and len(res) > 0:
            self.ring.dprint(f"InfernRTPIngest.run: res = {res}")

class InfernRTPIngest(InfernWrkThread):
    debug = True
    pkt_queue: Queue[Union[WIPkt,WIStreamUpdate]]
    def __init__(self):
        super().__init__()
        self.pkt_queue = Queue()
#        self.start()

    def dprint(self, *args):
        if self.debug:
            print(*args)

    def run(self):
        super().thread_started()
        self.dprint("InfernRTPIngest started")
        data, address, rtime = (None, None, None)
        while self.get_state() == RTPWrkTRun:
            wi = self.pkt_queue.get()
            if wi is None: break
            wi.stream._proc_in_tread(wi)
#        if data is not None:
#            self.dprint(f"InfernRTPIngest.run: last packet: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
#        self.dprint(f"InfernRTPIngest.run: exiting, total packets received: {npkts}")

    def stop(self):
        self.pkt_queue.put(None)
        super().stop()
        self.dprint("InfernRTPIngest stopped")

    def __del__(self):
        self.dprint("InfernRTPIngest.__del__")
