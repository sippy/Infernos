import torch
import soundfile as sf

from typing import Optional
import queue
import threading
from time import monotonic, sleep

from rtpsynth.RtpSynth import RtpSynth

from Core.Codecs.G711 import G711Codec

class TTSSMarkerGeneric():
    pass

class TTSSMarkerNewSent(TTSSMarkerGeneric):
    # This runs in the context of the RTPOutputWorker thread
    def on_proc(self, tro_self, *args): pass

class TTSSMarkerEnd(TTSSMarkerGeneric):
    pass

class AudioChunk():
    samplerate: int
    audio: torch.Tensor
    def __init__(self, audio, samplerate):
        self.audio = audio
        self.samplerate = samplerate

class RTPOutputStream():
    ptime: float = 0.0
    btime: Optional[float] = None
    nchunk: int = 0
    stime: Optional[float] = None
    itime: float
    ctime: Optional[float] = None
    def __init__(self, itime:float, device):
        self.itime = itime
        self.chunk = torch.empty(0).to(device)
        self.chunk_o = torch.empty(0).to(device)

    def chunk_in(self, chunk:AudioChunk, wrkr:'RTPOutputWorker'):
        self.nchunk += 1
        #print(chunk_n.size(), self.chunk.size())
        self.chunk = torch.cat((self.chunk, chunk.audio), dim=0)
        if self.ptime == 0.0:
            if self.btime == None:
                min_btime = 1.0
                self.btime = min(self.ctime - self.itime, min_btime)
                self.btime = chunk.samplerate * self.btime / 2
                self.btime = int(self.btime)
                if wrkr.debug:
                    print('self.btime', self.btime)
            if self.chunk.size(0) < self.btime:
                return None, f'{self.chunk.size(0)} < {self.btime}'
        if self.stime is None:
            self.stime = self.ctime
        return self.chunk, None

    def eos(self):
        self.chunk = self.chunk[:0]
        self.chunk_o = self.chunk_o[:0]
        self.ptime = 0.0
        self.stime = None
        self.itime = monotonic()

class RTPOutputWorker(threading.Thread):
    debug = True
    debug = False
    dl_ofname: str = None
    data_log = None
    pkg_send_f = None
    state_lock: threading.Lock = None
    frames_rcvd = 0
    frames_prcsd = 0
    has_ended = False
    codec: G711Codec
    samplerate_in: int
    samplerate_out: int = G711Codec.default_sr
    out_ft: int = 30 # ms

    def __init__(self, device, samplerate_in=G711Codec.default_sr):
        self.itime = monotonic()
        self.device = device
        #if os.path.exists(self.ofname):
        #    self.data, _ = sf.read(self.ofname)
        self.data_queue = queue.Queue()
        self.codec = G711Codec(samplerate_in).to(device)
        self.state_lock = threading.Lock()
        self.samplerate_in = samplerate_in
        super().__init__(target=self.consume_audio)
        self.daemon = True

    def enable_datalog(self, dl_ofname):
        self.dl_ofname = dl_ofname

    def set_pkt_send_f(self, pkt_send_f):
        self.pkt_send_f = pkt_send_f

    def ended(self):
        self.state_lock.acquire()
        t = self.has_ended
        self.state_lock.release()
        return t

    def end(self):
        self.state_lock.acquire()
        self.has_ended = True
        self.state_lock.release()

    def update_frm_ctrs(self, rcvd_inc=0, prcsd_inc=0):
        self.state_lock.acquire()
        self.frames_rcvd += rcvd_inc
        self.frames_prcsd += prcsd_inc
        self.state_lock.release()

    def get_frm_ctrs(self):
        self.state_lock.acquire()
        res = (self.frames_rcvd, self.frames_prcsd)
        self.state_lock.release()
        return res

    def soundout(self, chunk):
        #print(f'soundout: {monotonic():4.3f}')
        #return (0, False)
        ismark = isinstance(chunk, TTSSMarkerGeneric)
        iseos = isinstance(chunk, TTSSMarkerEnd)
        assert ismark or chunk.size(0) > 0
        if self.debug and not ismark:
            print(f'len(chunk) = {len(chunk)}')
        if not ismark:
            chunk = chunk.to(self.device)
        self.data_queue.put(chunk)
        if iseos:
            self.join()
        return (self.data_queue.qsize(), False)

    def consume_audio(self):
        out_pt = self.codec.pt
        out_fsize = int(self.samplerate_out * self.out_ft / 1000)
        pos = RTPOutputStream(self.itime, self.device)
        rsynth = RtpSynth(self.samplerate_out, self.out_ft)
        while not self.ended():
            try:
                chunk_n = self.data_queue.get(timeout=0.03)
            except queue.Empty:
                continue
            if isinstance(chunk_n, TTSSMarkerEnd):
                break
            if isinstance(chunk_n, TTSSMarkerNewSent):
                #pos.btime = None
                prcsd_inc=pos.chunk.size(0) + (pos.chunk_o.size(0) * 2)
                self.update_frm_ctrs(prcsd_inc=prcsd_inc)
                pos.eos()
                rsynth.resync()
                rsynth.set_mbt(1)
                chunk_n.on_proc(self)
                continue
            self.update_frm_ctrs(rcvd_inc=chunk_n.size(0))
            pos.ctime = monotonic()

            if self.dl_ofname is not None:
                if self.data_log is None:
                    self.data_log = chunk_n
                else:
                    self.data_log = torch.cat((self.data_log,
                                           chunk_n))
            achunk = AudioChunk(chunk_n, self.samplerate_in)
            chunk_o_n, explain = pos.chunk_in(achunk, self)
            if chunk_o_n is None:
                if self.debug: print(f'consume_audio({len(pos.chunk)}), {explain}')
                continue

            if self.samplerate_in != self.samplerate_out:
                sz = chunk_o_n.size(0)
                chunk_o_n = self.codec.resampler[0](chunk_o_n)
                assert chunk_o_n.size(0) == sz // (self.samplerate_in // self.samplerate_out)
            pos.chunk_o = torch.cat((pos.chunk_o, chunk_o_n), dim=0)

            etime = pos.ctime - pos.stime
            if self.debug:
                print(f'consume_audio({len(pos.chunk)}), etime = {etime}, pos.ptime = {pos.ptime}')

            pos.chunk = pos.chunk[:0]

            while pos.chunk_o.size(0) >= out_fsize:
                self.update_frm_ctrs(prcsd_inc=out_fsize*2)
                packet = pos.chunk_o[:out_fsize]
                pos.chunk_o = pos.chunk_o[out_fsize:]

                pos.ptime += len(packet) / self.samplerate_out
                etime = pos.ctime - pos.stime

                #print(packet.size())
                #packet = (packet * 20000).to(torch.int16)
                #packet = packet.byte().cpu().numpy()
                packet = self.codec.encode(packet, resample=False).cpu().numpy()
                #print('packet', packet.min(), packet.max(), packet[:10])
                packet = packet.tobytes()
                #print(len(packet), packet[:10])
                pkt = rsynth.next_pkt(out_fsize, out_pt, pload=packet)
                if self.pkt_send_f is not None:
                    self.pkt_send_f(pkt)
                #print(len(pkt))
                if self.debug:
                    print(f'consume_audio({len(pos.chunk_o)}), etime = {etime}, pos.ptime = {pos.ptime}')
                if self.ended():
                    break
                if pos.ptime > etime:
                    sleep(pos.ptime - etime)
                    if self.ended():
                        break
                    pos.ctime = monotonic()
                    if self.debug:
                        print(f'consume_audio, sleep({pos.ptime - etime})')

    def __del__(self):
        if self.debug:
            print('RTPOutputWorker.__del__')
        #self.worker_thread.join()
        if self.data_log is None:
            return
        amplification_dB = 20.0
        data = self.data_log #* (10 ** (amplification_dB / 20))
        sf.write(self.dl_ofname, data.detach().cpu().numpy(),
                 samplerate=self.samplerate_in)
