from typing import Optional, Union
from queue import Queue
from functools import partial

from rtpsynth.RtpJBuf import RtpJBuf, RTPFrameType, RTPParseError

from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun
from Core.VAD.ZlibVAD import ZlibVAD
from Core.Codecs.G711 import G711Codec
from RTP.AudioInput import AudioInput

class WIPkt():
    def __init__(self, stream: 'RTPInStream', data, address, rtime):
        self.stream = stream
        self.data = data
        self.address = address
        self.rtime = rtime

class WIStreamUpdate():
    def __init__(self, stream: 'RTPInStream'):
        self.stream = stream

class WIStreamConnect():
    def __init__(self, stream: 'RTPInStream', ain:AudioInput):
        self.stream = stream
        self.ain = ain

class RTPInStream():
    jb_size: int = 8
    input_sr: int = 8000
    last_output_lseq: Optional[int] = None
    vad: ZlibVAD
    codec: G711Codec
    output_sr: int = 16000
    npkts: int = 0
    ain: AudioInput
    def __init__(self, ring:'InfernRTPIngest', device:str):
        self.jbuf = RtpJBuf(self.jb_size)
        self.vad = ZlibVAD(self.input_sr)
        self.codec = G711Codec().to(device)
        self.ring = ring
        self.ain = AudioInput()

    def rtp_received(self, data, address, rtime):
        #self.dprint(f"InfernRTPIngest.rtp_received: len(data) = {len(data)}")
        self.ring.pkt_queue.put(WIPkt(self, data, address, rtime))

    def stream_update(self):
        self.ring.pkt_queue.put(WIStreamUpdate(self))

    def stream_connect(self, ain:AudioInput):
        self.ring.pkt_queue.put(WIStreamConnect(self, ain))

    def _proc_in_tread(self, wi:Union[WIPkt,WIStreamUpdate]):
        def dprint(msg:str): return self.ring.dprint(f'InfernRTPIngest.run: {msg}') if self.ring.debug else None

        if isinstance(wi, WIStreamUpdate):
            dprint("stream update")
            self.jbuf = RtpJBuf(self.jb_size)
            self.last_output_lseq = None
            return
        if isinstance(wi, WIStreamConnect):
            dprint("stream connect")
            self.ain = wi.ain
            return
        data, address, rtime = wi.data, wi.address, wi.rtime
        try:
            res = self.jbuf.udp_in(data)
        except RTPParseError as e:
            dprint(f"RTPParseError: {e}")
            return
        self.npkts += 1
        if self.npkts == 1:
            dprint(f"address={address}, rtime={rtime}, len(data) = {len(data)} data={data[:40]}")
        for pkt in res:
            if pkt.content.type == RTPFrameType.ERS: print(f"ERS packet received {pkt.content.lseq_start=}, {pkt.content.lseq_end=}")
            assert pkt.content.type != RTPFrameType.ERS
            if self.npkts < 10:
                dprint(f"{pkt.content.frame.rtp.lseq=}")
            assert self.last_output_lseq is None or pkt.content.frame.rtp.lseq == self.last_output_lseq + 1
            self.last_output_lseq = pkt.content.frame.rtp.lseq
            if self.npkts < 10:
                dprint(f"{len(pkt.rtp_data)=}, {type(pkt.rtp_data)=}")
            out = self.vad.ingest(pkt.rtp_data, self.audio_vad_chunk_in)
            if out is None: continue
            chunk = self.codec.decode(out.chunk, resample=True, sample_rate=self.output_sr)
            dprint(f"active chunk: {len(chunk.audio)=}")
            if self.ain.vad_chunk_in is not None:
                self.ain.vad_chunk_in(chunk=chunk)
        if self.npkts < 10 and len(res) > 0:
            dprint(f"{res=}")

    def audio_vad_chunk_in(self, chunk_data:bytes, active:bool):
        if self.ain.audio_in is None: return
        chunk = self.codec.decode(chunk_data, resample=True, sample_rate=self.output_sr)
        self.ain.audio_in(chunk=chunk)

class InfernRTPIngest(InfernWrkThread):
    debug = False
    pkt_queue: Queue[Union[WIPkt,WIStreamUpdate,WIStreamConnect]]
    _start_queue: Queue[int]
    def __init__(self):
        super().__init__()
        self.pkt_queue = Queue()

    def start(self):
        self._start_queue = Queue()
        super().start()
        self._start_queue.get()
        del self._start_queue

    def dprint(self, *args):
        if self.debug:
            print(*args)

    def run(self):
        super().thread_started()
        self._start_queue.put(0)
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
