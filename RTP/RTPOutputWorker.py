from typing import Optional, Dict, Union
from fractions import Fraction
import queue
import threading
from time import monotonic, sleep

import ray
from rtpsynth.RtpSynth import RtpSynth
import torch
import soundfile as sf

from config.InfernGlobals import InfernGlobals as IG
from Core.Codecs.G711 import G711Codec
from Core.AudioChunk import AudioChunk
from Core.OutputMuxer import OutputMTMuxer
from Core.AStreamMarkers import ASMarkerGeneric

class RTPOutputWorker(threading.Thread):
    data_queue: queue.Queue[Union[AudioChunk, ASMarkerGeneric]]
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
    out_ft: int # in ms

    def __init__(self, device, out_ft):
        self.itime = monotonic()
        self.device = device
        #if os.path.exists(self.ofname):
        #    self.data, _ = sf.read(self.ofname)
        self.data_queue = queue.Queue()
        self.codec = G711Codec().to(device)
        self.state_lock = threading.Lock()
        self.out_ft = out_ft
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

    def soundout(self, chunk:Union[AudioChunk, ASMarkerGeneric]):
        #print(f'soundout: {monotonic():4.3f}')
        #return (0, False)
        ismark = isinstance(chunk, ASMarkerGeneric)
        assert ismark or chunk.audio.size(0) > 0
        if (self.debug or chunk.debug) and not ismark:
            print(f'len(chunk) = {len(chunk.audio)}')
        if not ismark:
            chunk.audio = chunk.audio.to(self.device)
        self.data_queue.put(chunk)
        return (self.data_queue.qsize(), False)

    def consume_audio(self):
        out_pt = self.codec.pt
        out_fsize = self.samplerate_out * self.out_ft // 1000
        ptime = Fraction(0)
        stime = None
        rsynth = RtpSynth(self.samplerate_out, self.out_ft)
        qtimeout = Fraction(self.out_ft, 1000)
        out_qsize = self.out_ft * (self.samplerate_out // 10 // self.out_ft) # ~0.1 sec (rounded to a frame size)
        mix = OutputMTMuxer(self.samplerate_out, out_qsize, self.device)
        while not self.ended():
            ctime = monotonic()
            try:
                chunk_n = self.data_queue.get(block=False)
            except queue.Empty:
                chunk_o_n = mix.idle(self)
                if chunk_o_n is None:
                    if stime is not None:
                        ptime += qtimeout
                        etime = ctime - stime
                        if ptime > etime:
                            sleep(ptime - etime)
                        if self.debug: print(f'{self}.consume_audio, skip {ptime - etime=}')
                        rsynth.skip(1)
                    else:
                        sleep(float(qtimeout))
                    continue
            else:
                if isinstance(chunk_n, AudioChunk): self.update_frm_ctrs(rcvd_inc=chunk_n.audio.size(0))
                mix.chunk_in(chunk_n)
                continue

            if stime is None:
                stime = ctime

            while chunk_o_n.size(0) >= out_fsize:
                self.update_frm_ctrs(prcsd_inc=out_fsize*2)
                packet = chunk_o_n[:out_fsize]
                chunk_o_n = chunk_o_n[out_fsize:]

                ptime += Fraction(len(packet), self.samplerate_out)
                etime = ctime - stime

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
                if chunk_n.debug or self.debug:
                    print(f'{self}.consume_audio({etime=}, {ptime=}')
                if self.ended():
                    break
                if ptime > etime:
                    sleep(ptime - etime)
                    if self.ended():
                        break
                    ctime = monotonic()
                    if chunk_n.debug or self.debug:
                        print(f'consume_audio, sleep({ptime - etime})')
            #if done_cb is not None:
            #    rsynth.resync()
            #    rsynth.set_mbt(1)
            #    ptime = 0.0
            #    stime = None
            #    done_cb(self)

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
