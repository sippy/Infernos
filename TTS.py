from datasets import load_dataset
try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    ipex = None
import torch
import soundfile as sf
import os.path
import numpy as np
import torchaudio.transforms as T

from HelloSippyTTSRT.HelloSippyRT import HelloSippyRT

import queue
import threading
from time import monotonic, sleep

from scipy.signal import butter, freqz, firwin
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from rtpsynth.RtpSynth import RtpSynth

import audioop

def get_LPF(fs = 16000, cutoff = 4000.0):
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b
    order = 6
    filter_kernel = butter_lowpass(cutoff, fs, order)
    filter_kernel = torch.tensor(filter_kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('xpu')
    return filter_kernel

@torch.no_grad()
def get_PBF0(fs = 16000, l_cut = 50.0, h_cut = 4000.0):
    # Create a simple PyTorch model with a single Conv1d layer
    class PassBandFilter(nn.Module):
        def __init__(self, coefficients):
            super(PassBandFilter, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=len(coefficients), padding=len(coefficients)//2, bias=False)

            # Load the filter coefficients into the Conv1d layer
            self.conv1.weight.data = torch.Tensor(coefficients).float().view(1, 1, -1)

        def forward(self, x):
            return self.conv1(x)

    # Design a Butterworth pass-band filter
    order = 6  # Order of the filter

    b, a = butter(order, [l_cut, h_cut], btype='band', fs=fs)
    w, h = freqz(b, a, fs=fs)

    center_freq = np.sqrt(l_cut * h_cut)
    w_center = 2 * np.pi * center_freq / fs
    gain_at_center = np.abs(np.polyval(b, np.exp(1j * w_center)) / np.polyval(a, np.exp(1j * w_center)))

    b /= gain_at_center

    # Initialize the PyTorch model
    return PassBandFilter(b).to('xpu')

def get_PBF(fs = 16000, l_cut = 75.0, h_cut = 4000.0):
    numtaps = 1024  # Number of filter taps (coefficients)
    coeffs = firwin(numtaps, [l_cut, h_cut], pass_zero='bandpass', fs=fs)

    # Convert to PyTorch tensor
    filter_kernel = torch.tensor(coeffs, dtype=torch.float32).view(1, 1, -1).to('xpu')
    return filter_kernel

def numpy_audioop_helper(x, xdtype, func, width, ydtype):
    '''helper function for using audioop buffer conversion in numpy'''
    xi = np.asanyarray(x).astype(xdtype)
    if np.any(x != xi):
        xinfo = np.iinfo(xdtype)
        raise ValueError("input must be %s [%d..%d]" % (xdtype, xinfo.min, xinfo.max))
    y = np.frombuffer(func(xi.tobytes(), width), dtype=ydtype)
    return y.reshape(xi.shape)

def audioop_ulaw_compress(x):
    return numpy_audioop_helper(x, np.int16, audioop.lin2ulaw, 2, np.uint8)

class TTSSMarkerGeneric():
    pass

class TTSSMarkerNewSent(TTSSMarkerGeneric):
    pass

class TTSSMarkerEnd(TTSSMarkerGeneric):
    pass


TSO_SESSEND = None

class TTSSoundOutput(threading.Thread):
    pre_frames = 2
    _frame_size = 256
    debug = False
    dl_ofname: str
    data_log = None
    o_flt = None
    num_mel_bins = None
    pkg_send_f = None
    dl_ofname = None
    state_lock: threading.Lock = None
    frames_rcvd = 0
    frames_prcsd = 0
    has_ended = False

    def __init__(self, num_mel_bins, device, vocoder=None, filter_out=False):
        self.itime = monotonic()
        self.vocoder = vocoder
        self.num_mel_bins = num_mel_bins
        self.device = device
        #if os.path.exists(self.ofname):
        #    self.data, _ = sf.read(self.ofname)
        self.data_queue = queue.Queue()
        self.samplerate = 16000
        if filter_out:
            self.o_flt = get_PBF(self.samplerate)
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
        codec = T.MuLawEncoding().to(self.device)
        nchunk = 0
        btime = None
        chunk = torch.empty(0).to(self.device)
        prev_chunk_len = 0
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
            #print('chunk_o_n', chunk_o_n.min(), chunk_o_n.max(), chunk_o_n.float().mean())
            ##_chunk_o_n = (chunk_o_n * 32768).to(torch.int16)
            ##print(_chunk_o_n[:10])
            ##print(audioop_ulaw_compress(_chunk_o_n.cpu().numpy())[:10])
            ##chunk_o_n = codec(chunk_o_n)
            #print('chunk_o_n', chunk_o_n.min(), chunk_o_n.max(), chunk_o_n.float().mean())
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
                packet = (packet * 20000).to(torch.int16)
                #packet = packet.byte().cpu().numpy()
                packet = audioop_ulaw_compress(packet.cpu().numpy())
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
        print('__del__')
        #self.worker_thread.join()
        if self.o_flt is not None:
            print(self.o_flt.size(2))
        if self.data_log is None:
            return
        amplification_dB = 20.0
        data = self.data_log #* (10 ** (amplification_dB / 20))
        #data = self.o_flt(data.unsqueeze(0).unsqueeze(0)).squeeze()
        if self.o_flt is not None:
            data = F.conv1d(data.unsqueeze(0).unsqueeze(0),
                            self.o_flt,
                            padding=(self.o_flt.size(2) - 1) // 2)
            data = data.squeeze()
        sf.write(self.dl_ofname, data.detach().cpu().numpy(),
                 samplerate=self.samplerate)

class TTS(HelloSippyRT):
    device = 'cuda' if ipex is None else 'xpu'

    def __init__(self):
        super().__init__(self.device)
        if ipex is not None:
            self.model = ipex.optimize(self.model)
            self.vocoder = ipex.optimize(self.vocoder)
            self.chunker = ipex.optimize(self.chunker)
            #raise Exception(f"{type(hsrt.chunker)}")

    def dotts(self, text, ofname):
        if False:
            tts_voc, so_voc = None, self.vocoder
        else:
            tts_voc, so_voc = self.vocoder, None
        writer = TTSSoundOutput(self.model.config.num_mel_bins,
                                self.device,
                                vocoder=so_voc)
        writer.enable_datalog(ofname)
        writer.start()

        speaker_embeddings = self.hsrt.get_rand_voice()

        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        speech = self.generate_speech_rt(inputs["input_ids"], writer.soundout,
                                               speaker_embeddings,
                                               vocoder=tts_voc)
        writer.soundout(TTSSMarkerEnd())

    def get_pkt_proc(self):
        writer = TTSSoundOutput(0, self.device)
        return writer


if __name__ == '__main__':
    tts = TTS()
    prompts = (
        "Hello and welcome to Sippy Software, your VoIP solution provider.",
        "Today is Wednesday twenty third of August two thousand twenty three, five thirty in the afternoon.",
        "For many applications, such as sentiment analysis and text summarization, pretrained models work well without any additional model training.",
        "This message has been generated by combination of the Speech tee five pretrained text to speech models by Microsoft and fine tuned Hello Sippy realtime vocoder by Sippy Software Inc.",
        )
    for i, prompt in enumerate(prompts):
        print(i)
        fname = f"tts_example{i}.wav"
#    prompt = "Hello and welcome to Sippy Software, your VoIP solution provider. Today is Wednesday twenty-third of August two thousand twenty three, five thirty in the afternoon."
#    prompt = 'For many applications, such as sentiment analysis and text summarization, pretrained models work well without any additional model training.'
    #prompt = "I've also played with the text-to-speech transformers those are great actually. I have managed making API more realtime and it works nicely!"
        tts.dotts(prompt, fname)
