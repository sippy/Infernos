from typing import Optional
from queue import Queue, Empty as QueueEmpty

from rtpsynth.RtpJBuf import RtpJBuf, RTPFrameType, RTPParseError

from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun
from Core.VAD.ZlibVAD import ZlibVAD
from Core.Codecs.G711 import G711Codec

class InfernRTPIngest(InfernWrkThread):
    debug = True
    pkt_queue: Queue = None
    jb_size: int = 8
    input_sr: int = 8000
    output_sr: int
    codec: str = "PCMU"
    last_output_lseq: Optional[int] = None
    vad: ZlibVAD
    codec: G711Codec
    def __init__(self, chunk_in, device, output_sr: int = 16000):
        super().__init__()
        self.pkt_queue = Queue()
        self.output_sr = output_sr
        self.vad = ZlibVAD(self.input_sr)
        self.chunk_in = chunk_in
        self.codec = G711Codec(self.output_sr).to(device)
#        self.start()

    def dprint(self, *args):
        if self.debug:
            print(*args)

    def run(self):
        super().thread_started()
        self.dprint("InfernRTPIngest started")
        data, address, rtime = (None, None, None)
        npkts = 0
        jbuf = RtpJBuf(self.jb_size)
        while self.get_state() == RTPWrkTRun:
            try:
                data, address, rtime = self.pkt_queue.get(timeout=0.03)
            except QueueEmpty:
                continue
            try:
                res = jbuf.udp_in(data)
            except RTPParseError as e:
                self.dprint(f"InfernRTPIngest.run: RTPParseError: {e}")
                continue
            npkts += 1
            if npkts == 1:
                self.dprint(f"InfernRTPIngest.run: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
            for pkt in res:
                assert pkt.content.type != RTPFrameType.ERS
                if npkts < 10:
                    self.dprint(f"InfernRTPIngest.run: {pkt.content.frame.rtp.lseq=}")
                assert self.last_output_lseq is None or pkt.content.frame.rtp.lseq == self.last_output_lseq + 1
                self.last_output_lseq = pkt.content.frame.rtp.lseq
                if npkts < 10:
                    self.dprint(f"InfernRTPIngest.run: {len(pkt.rtp_data)=}, {type(pkt.rtp_data)=}")
                out = self.vad.ingest(pkt.rtp_data)
                if out is None: continue
                chunk = self.codec.decode(out.chunk)
                self.dprint(f"InfernRTPIngest.run: active chunk: {len(chunk)=}")
                self.chunk_in(chunk)
            if npkts < 10 and len(res) > 0:
                self.dprint(f"InfernRTPIngest.run: res = {res}")
        if data is not None:
            self.dprint(f"InfernRTPIngest.run: last packet: address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
        self.dprint(f"InfernRTPIngest.run: exiting, total packets received: {npkts}")

    def stop(self):
        super().stop()
        self.dprint("InfernRTPIngest stopped")
        del self.chunk_in

    def rtp_received(self, data, address, udp_server, rtime):
        #self.dprint(f"InfernRTPIngest.rtp_received: len(data) = {len(data)}")
        self.pkt_queue.put((data, address, rtime))

    def __del__(self):
        self.dprint("InfernRTPIngest.__del__")
