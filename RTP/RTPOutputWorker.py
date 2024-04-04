from typing import Optional, Dict, Union
import queue
import threading
from time import monotonic, sleep

import ray
from rtpsynth.RtpSynth import RtpSynth
import torch
import soundfile as sf

from Core.Codecs.G711 import G711Codec
from Core.AudioChunk import AudioChunk

class TTSSMarkerGeneric():
    track_id: int = 0

class TTSSMarkerNewSent(TTSSMarkerGeneric):
    # This runs in the context of the RTPOutputWorker thread
    def on_proc(self, tro_self, *args): pass

class TTSSMarkerEnd(TTSSMarkerGeneric):
    pass

class TTSSMarkerSentDoneCB(TTSSMarkerNewSent):
    def __init__(self, done_cb:callable, sync:bool=False):
        super().__init__()
        self.done_cb = done_cb
        self.sync = sync

    def on_proc(self, tro_self):
        print(f'{monotonic():4.3f}: TTSSMarkerSentDoneCB.on_proc')
        x = self.done_cb()
        if self.sync: ray.get(x)

class RTPOutputStream():
    ptime: float = 0.0
    btime: Optional[float] = None
    nchunk: int = 0
    stime: Optional[float] = None
    itime: float
    ctime: Optional[float] = None
    tracks_in: Dict[str, torch.Tensor]
    def __init__(self, itime:float, device):
        self.itime = itime
        self.tracks_in = {}
        self.chunk_o = torch.empty(0).to(device)

    def new_track(self, track_id:int):
        self.tracks_in[track_id] = torch.empty(0)

    def chunk_in(self, chunk:AudioChunk, wrkr:'RTPOutputWorker'):
        self.nchunk += 1
        if chunk.track_id not in self.tracks_in: self.new_track(chunk.track_id)
        previous = self.tracks_in[chunk.track_id]
        merged = torch.cat((previous, chunk.audio), dim=0)
        if wrkr.debug: print(f'chunk_in[{chunk.track_id}]: {chunk.audio.size()=}, {merged.size()=}')
        self.tracks_in[chunk.track_id] = merged
        if self.ptime == 0.0:
            if self.btime == None:
                min_btime = 1.0
                self.btime = min(self.ctime - self.itime, min_btime)
                self.btime = chunk.samplerate * self.btime / 2
                self.btime = int(self.btime)
                if wrkr.debug:
                    print('self.btime', self.btime)
            if merged.size(0) < self.btime:
                return None, f'{previous.size(0)=}, {merged.size(0)=} < {self.btime=}'
        if self.stime is None:
            self.stime = self.ctime
        if len(self.tracks_in) > 1:
            self.tracks_in[chunk.track_id] = merged
            rdylen = min(t.size(0) for t in self.tracks_in.values())
            if rdylen == 0: return None, 'some channels are not ready to proceed yet'
            mixin = []
            for tid, rdy, nrdy in [(tid, t[:rdylen], t[rdylen:]) for tid, t in self.tracks_in.items()]:
                self.tracks_in[tid] = nrdy
                mixin.append(rdy)
            merged = torch.sum(torch.stack(mixin), dim=0)
            max_val = torch.max(torch.abs(merged))
            if max_val > 1:
                merged /= max_val
        else:
            self.tracks_in[chunk.track_id] = previous[:0]
        return merged, None

    def eos(self, track_id:int):
        del self.tracks_in[track_id]
        if len(self.tracks_in) != 0:
            return False
        self.chunk_o = self.chunk_o[:0]
        self.ptime = 0.0
        self.stime = None
        self.itime = monotonic()
        return True

    def get_buf_nframes(self):
        return sum(c.size(0) for c in self.tracks_in.values()) + (self.chunk_o.size(0) * 2)

class RTPOutputWorker(threading.Thread):
    data_queue: queue.Queue[Union[AudioChunk, TTSSMarkerGeneric]]
    debug = False
    dl_ofname: str = None
    data_log = None
    pkg_send_f = None
    state_lock: threading.Lock = None
    frames_rcvd = 0
    frames_prcsd = 0
    has_ended = False
    codec: G711Codec
    samplerate_out: int = G711Codec.default_sr
    out_ft: int = 30 # ms

    def __init__(self, device):
        self.itime = monotonic()
        self.device = device
        #if os.path.exists(self.ofname):
        #    self.data, _ = sf.read(self.ofname)
        self.data_queue = queue.Queue()
        self.codec = G711Codec().to(device)
        self.state_lock = threading.Lock()
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

    def soundout(self, chunk:Union[AudioChunk, TTSSMarkerGeneric]):
        #print(f'soundout: {monotonic():4.3f}')
        #return (0, False)
        ismark = isinstance(chunk, TTSSMarkerGeneric)
        iseos = isinstance(chunk, TTSSMarkerEnd)
        assert ismark or chunk.audio.size(0) > 0
        if self.debug and not ismark:
            print(f'len(chunk) = {len(chunk.audio)}')
        if not ismark:
            chunk.audio = chunk.audio.to(self.device)
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
                self.update_frm_ctrs(prcsd_inc=pos.get_buf_nframes())
                if pos.eos(chunk_n.track_id):
                    rsynth.resync()
                    rsynth.set_mbt(1)
                chunk_n.on_proc(self)
                continue
            self.update_frm_ctrs(rcvd_inc=chunk_n.audio.size(0))
            pos.ctime = monotonic()

            if chunk_n.samplerate != self.samplerate_out:
                sz = chunk_n.audio.size(0)
                resampler = self.codec.get_resampler(chunk_n.samplerate, self.samplerate_out)
                chunk_n.audio = resampler(chunk_n.audio)
                assert chunk_n.audio.size(0) == sz // (chunk_n.samplerate // self.samplerate_out)
                chunk_n.samplerate = self.samplerate_out

            if self.dl_ofname is not None:
                a = chunk_n.audio
                if self.data_log is None:
                    self.data_log = a
                else:
                    self.data_log = torch.cat((self.data_log, a))

            chunk_o_n, explain = pos.chunk_in(chunk_n, self)
            if chunk_o_n is None:
                if self.debug: print(f'consume_audio({explain}')
                continue

            pos.chunk_o = torch.cat((pos.chunk_o, chunk_o_n), dim=0)

            etime = pos.ctime - pos.stime
            #if self.debug:
            #    print(f'consume_audio({len(pos.chunk)}), etime = {etime}, pos.ptime = {pos.ptime}')

            while pos.chunk_o.size(0) >= out_fsize:
                self.update_frm_ctrs(prcsd_inc=out_fsize*2)
                packet = pos.chunk_o[:out_fsize]
                pos.chunk_o = pos.chunk_o[out_fsize:]

                pos.ptime += len(packet) / self.samplerate_out
                etime = pos.ctime - pos.stime

                #print(packet.size())
                #packet = (packet * 20000).to(torch.int16)
                #packet = packet.byte().cpu().numpy()
                packet = self.codec.encode(packet).cpu().numpy()
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
                 samplerate=self.samplerate_out)
