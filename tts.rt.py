from transformers import SpeechT5Processor, SpeechT5HifiGanConfig, SpeechT5HifiGan
#from HIFIGanVocoder import HIFIGanVocoder
from datasets import load_dataset
import intel_extension_for_pytorch as ipex
import torch
import soundfile as sf
import os.path
import numpy as np
from waveglow_vocoder import WaveGlowVocoder
import torchaudio.transforms as T

from HelloSippyTTSRT.HelloSippyRT import HelloSippyRT as SpeechT5ForTextToSpeech

def load_PWG(device):
    from parallel_wavegan.utils import download_pretrained_model
    from parallel_wavegan.utils import load_model
    import yaml

    download_pretrained_model("arctic_slt_parallel_wavegan.v1")
    vocoder_conf = "ParallelWaveGAN/egs/arctic/voc1/conf/parallel_wavegan.v1.yaml"
    with open(vocoder_conf) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return load_model(".cache/parallel_wavegan/arctic_slt_parallel_wavegan.v1/checkpoint-400000steps.pkl", config).to(device)

import queue
import threading
from time import monotonic, sleep

from scipy.signal import butter, freqz, firwin
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

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

class TTSSoundOutput():
    pre_frames = 2
    _frame_size = 256
    debug = False
    ofname: str
    data = None
    o_flt = None
    num_mel_bins = None

    def __init__(self, ofname, num_mel_bins, device, vocoder = None, filter_out=False):
        self.ofname = ofname
        self.vocoder = vocoder
        self.num_mel_bins = num_mel_bins
        self.device = device
        #if os.path.exists(self.ofname):
        #    self.data, _ = sf.read(self.ofname)
        self.data_queue = queue.Queue()
        self.samplerate = 16000
        if filter_out:
            self.o_flt = get_PBF(self.samplerate)
        self.worker_thread = threading.Thread(target=self.consume_audio)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def soundout(self, chunk):
        assert chunk is None or chunk.size(0) > 0
        if self.debug and chunk is not None:
            print(f'len(chunk) = {len(chunk)}')
        self.data_queue.put(chunk)
        if chunk is None:
            self.worker_thread.join()
        return (self.data_queue.qsize(), False)

    def consume_audio(self):
        stime = ctime = None
        out_sr = 8000
        out_ft = 30
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
        while True:
            try:
                chunk = self.data_queue.get(timeout=0.03)
            except queue.Empty:
                continue
            if chunk is None:
                break
            ctime = monotonic()

            if self.vocoder:
                chunk = torch.cat((pfs, chunk), dim=0)
                outputs = self.vocoder(chunk).squeeze(0)[pf_trim:]
                #print(chunk.shape, outputs.shape)
                pfs = chunk[-self.pre_frames:, :]
                #print(pfs.size())
                chunk = outputs
            #chunk = chunk.cpu().numpy()

            if self.data is None:
                self.data = chunk
            else:
                self.data = torch.cat((self.data, chunk))
            if stime is None:
                stime = ctime

            chunk = resampler(chunk)
            chunk = codec(chunk)

            while len(chunk) >= out_fsize:
                packet = chunk[:out_fsize]
                chunk = chunk[out_fsize:]

                ptime += len(packet) / out_sr
                etime = ctime - stime
                print(f'consume_audio({len(chunk)}), etime = {etime}, ptime = {ptime}')
                if ptime > etime:
                    sleep(ptime - etime)
                    ctime = monotonic()
                    print(f'consume_audio, sleep({ptime - etime})')

    def __del__(self):
        print('__del__')
        #self.worker_thread.join()
        if self.o_flt is not None:
            print(self.o_flt.size(2))
        if self.data is not None:
            amplification_dB = 20.0
            data = self.data #* (10 ** (amplification_dB / 20))
            #data = self.o_flt(data.unsqueeze(0).unsqueeze(0)).squeeze()
            if self.o_flt is not None:
                data = F.conv1d(data.unsqueeze(0).unsqueeze(0),
                                self.o_flt,
                                padding=(self.o_flt.size(2) - 1) // 2)
                data = data.squeeze()
            sf.write(self.ofname, data.detach().cpu().numpy(),
                     samplerate=self.samplerate)

from utils import load_checkpoint, scan_checkpoint

class TTS():
    half_p = False
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to('xpu')
    if half_p:
        model = model.half()
    model.eval()
    model = ipex.optimize(model)
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    _vc_conf = SpeechT5HifiGanConfig()
    #_vc_conf.padding = 0
    #vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", config = _vc_conf).to(model.device)
    vocoder = SpeechT5HifiGan(config = _vc_conf).to(model.device)
    checkpoint_path = 'cp_hifigan.test'
    cp_g = scan_checkpoint(checkpoint_path, 'g_')
    state_dict_g = load_checkpoint(cp_g, model.device)
    vocoder.load_state_dict(state_dict_g['generator'])
    del state_dict_g
    if half_p:
        vocoder = vocoder.half()
    vocoder.eval()
    vocoder = ipex.optimize(vocoder)

#    vocoder = load_PWG(model.device)
#    vocoder = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')
#    vocoder = WaveGlowVocoder()
#    vocoder = HIFIGanVocoder(device = model.device)
#    vocoder = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft", map_location=torch.device(model.device))
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(model.device)

    def dotts(self, text, ofname):
        writer = TTSSoundOutput(ofname, self.model.config.num_mel_bins,
                                self.model.device) 
        speaker_embeddings = torch.randn(1, 512, device = self.model.device)
        speaker_embeddings = self.speaker_embeddings
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        speech = self.model.generate_speech_rt(inputs["input_ids"], writer.soundout,
                                               speaker_embeddings,
                                               vocoder=self.vocoder)

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
