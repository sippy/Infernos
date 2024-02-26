import torch
import soundfile as sf
import torchaudio.transforms as T

import audioop

import queue
import threading
from time import monotonic, sleep

from rtpsynth.RtpSynth import RtpSynth

class TTSSMarkerGeneric():
    pass

class TTSSMarkerNewSent(TTSSMarkerGeneric):
    # This runs in the context of the TTSRTPOutput thread
    def on_proc(self, tro_self, *args): pass

class TTSSMarkerEnd(TTSSMarkerGeneric):
    pass

ulaw_ct = torch.zeros(65536, dtype=torch.uint8)
for i in range(-32768, 32768):
    pcm_data = i.to_bytes(2, 'little', signed=True)
    ulaw_data = audioop.lin2ulaw(pcm_data, 2)
    ulaw_value = ulaw_data[0]  # Get the byte value from bytes
    ulaw_ct[i + 32768] = ulaw_value  # Shift index to make it non-negative

def float_to_ulaw(audio_tensor):
    # Scale from [-1, 1] to [-32768, 32767]
    audio_scaled = torch.clamp(audio_tensor * 32767.0, -32768, 32767).to(torch.int16)

    # Shift and look up in the conversion table
    audio_ulaw = ulaw_ct[(audio_scaled + 32768).long()]

    return audio_ulaw

class TTSRTPOutput(threading.Thread):
    debug = True
    pre_frames = 2
    _frame_size = 256
    debug = False
    dl_ofname: str = None
    data_log = None
    num_mel_bins = None
    pkg_send_f = None
    state_lock: threading.Lock = None
    frames_rcvd = 0
    frames_prcsd = 0
    has_ended = False

    def __init__(self, num_mel_bins, device, vocoder=None):
        global ulaw_ct
        if ulaw_ct.device != device:
            ulaw_ct = ulaw_ct.to(device)
        self.itime = monotonic()
        self.vocoder = vocoder
        self.num_mel_bins = num_mel_bins
        self.device = device
        #if os.path.exists(self.ofname):
        #    self.data, _ = sf.read(self.ofname)
        self.data_queue = queue.Queue()
        self.samplerate = 16000
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

    def soundout(self, chunk):
        #print(f'soundout: {monotonic():4.3f}')
        #return (0, False)
        ismark = isinstance(chunk, TTSSMarkerGeneric)
        iseos = isinstance(chunk, TTSSMarkerEnd)
        assert ismark or chunk.size(0) > 0
        if self.debug and not ismark:
            print(f'len(chunk) = {len(chunk)}')
        self.data_queue.put(chunk)
        if iseos:
            self.join()
        return (self.data_queue.qsize(), False)

    def consume_audio(self):
        itime = self.itime
        stime = ctime = None
        out_sr = 8000
        out_ft = 30
        out_pt = 0 # G.711u
        out_fsize = int(out_sr * out_ft / 1000)
        ptime = 0.0
        if self.vocoder:
            pfs = torch.zeros(self.pre_frames,
                              self.num_mel_bins,
                              device=self.vocoder.device)
            pf_trim = self.pre_frames * self._frame_size
        resampler = T.Resample(orig_freq=self.samplerate,
                               new_freq=out_sr
                               ).to(self.device)
        nchunk = 0
        btime = None
        chunk = torch.empty(0).to(self.device)
        chunk_o = torch.empty(0).to(self.device)
        rsynth = RtpSynth(out_sr, out_ft)
        while not self.ended():
            try:
                chunk_n = self.data_queue.get(timeout=0.03)
            except queue.Empty:
                continue
            if isinstance(chunk_n, TTSSMarkerEnd):
                break
            if isinstance(chunk_n, TTSSMarkerNewSent):
                #btime = None
                prcsd_inc=chunk.size(0) + (chunk_o.size(0) * 2)
                self.update_frm_ctrs(prcsd_inc=prcsd_inc)
                chunk = chunk[:0]
                chunk_o = chunk_o[:0]
                ptime = 0.0
                stime = None
                itime = monotonic()
                rsynth.resync()
                rsynth.set_mbt(1)
                chunk_n.on_proc(self)
                continue
            self.update_frm_ctrs(rcvd_inc=chunk_n.size(0))
            ctime = monotonic()

            if self.dl_ofname is not None:
                if self.data_log is None:
                    self.data_log = chunk_n
                else:
                    self.data_log = torch.cat((self.data_log,
                                           chunk_n))

            nchunk += 1
            #print(chunk_n.size(), chunk.size())
            chunk = torch.cat((chunk, chunk_n), dim=0)
            if ptime == 0.0:
                if btime == None:
                    min_btime = 1.0
                    btime = min(ctime - itime, min_btime)
                    btime = self.samplerate * btime / 2
                    if self.vocoder:
                        btime /= self._frame_size
                    btime = int(btime)
                    if self.debug:
                        print('btime', btime)
                if chunk.size(0) < btime:
                    if self.debug:
                        print(f'{chunk.size(0)} < {btime}')
                    continue

            if self.vocoder:
                chunk_o_n = torch.cat((pfs, chunk), dim=0)
                outputs = self.vocoder(chunk_o_n).squeeze(0)[pf_trim:]
                #print(chunk.shape, outputs.shape)
                pfs = chunk_o_n[-self.pre_frames:, :]
                #print(pfs.size())
                chunk_o_n = outputs
            else:
                chunk_o_n = chunk
            #chunk = chunk.cpu().numpy()

            if stime is None:
                stime = ctime

            sz = chunk_o_n.size(0)
            chunk_o_n = resampler(chunk_o_n)
            assert chunk_o_n.size(0) == sz / 2
            chunk_o = torch.cat((chunk_o, chunk_o_n), dim=0)

            etime = ctime - stime
            if self.debug:
                print(f'consume_audio({len(chunk)}), etime = {etime}, ptime = {ptime}')

            chunk = chunk[:0]

            while chunk_o.size(0) >= out_fsize:
                self.update_frm_ctrs(prcsd_inc=out_fsize*2)
                packet = chunk_o[:out_fsize]
                chunk_o = chunk_o[out_fsize:]

                ptime += len(packet) / out_sr
                etime = ctime - stime

                #print(packet.size())
                #packet = (packet * 20000).to(torch.int16)
                #packet = packet.byte().cpu().numpy()
                packet = float_to_ulaw(packet).cpu().numpy()
                #print('packet', packet.min(), packet.max(), packet[:10])
                packet = packet.tobytes()
                #print(len(packet), packet[:10])
                pkt = rsynth.next_pkt(out_fsize, out_pt, pload=packet)
                if self.pkt_send_f is not None:
                    self.pkt_send_f(pkt)
                #print(len(pkt))
                if self.debug:
                    print(f'consume_audio({len(chunk_o)}), etime = {etime}, ptime = {ptime}')
                if self.ended():
                    break
                if ptime > etime:
                    sleep(ptime - etime)
                    if self.ended():
                        break
                    ctime = monotonic()
                    if self.debug:
                        print(f'consume_audio, sleep({ptime - etime})')

    def __del__(self):
        if self.debug:
            print('TTSRTPOutput.__del__')
        #self.worker_thread.join()
        if self.data_log is None:
            return
        amplification_dB = 20.0
        data = self.data_log #* (10 ** (amplification_dB / 20))
        sf.write(self.dl_ofname, data.detach().cpu().numpy(),
                 samplerate=self.samplerate)
