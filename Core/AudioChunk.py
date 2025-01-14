import torch
import requests
import soundfile as sf
from io import BytesIO

from config.InfernGlobals import InfernGlobals as IG

class AudioChunk():
    debug: bool = False
    samplerate: int
    audio:torch.Tensor
    track_id: int = 0
    active: bool = True
    def __init__(self, audio:torch.Tensor, samplerate:int):
        assert isinstance(audio, torch.Tensor)
        self.audio = audio
        self.samplerate = samplerate

    def resample(self, sample_rate:int):
        assert sample_rate != self.samplerate
        audio = self.audio.to(torch.float)
        audio = IG.get_resampler(self.samplerate, sample_rate, audio.device)(audio).to(self.audio.dtype)
        self.samplerate, self.audio = sample_rate, audio
        return self

    def duration(self):
        return self.audio.size(0) / self.samplerate

class VadAudioChunk(AudioChunk):
    debug: bool = False
    ipos: int
    def __init__(self, audio:torch.Tensor, samplerate:int, ipos:int):
        super().__init__(audio, samplerate)
        self.ipos = ipos

    def tpos(self):
        return self.ipos / self.samplerate

    def append(self, other:'VadAudioChunk'):
        assert self.samplerate == other.samplerate
        if self.debug:
            print(f'VadAudioChunk.append: {self.ipos=} {self.audio.size(0)=} {other.ipos=} {other.audio.size(0)=}')
        sdiff = other.ipos - (self.ipos + self.audio.size(0))
        assert sdiff >= 0
        if sdiff > 0:
            self.audio = torch.cat((self.audio, torch.zeros(sdiff, dtype=self.audio.dtype, device=self.audio.device)), dim=0)
        self.audio = torch.cat((self.audio, other.audio), dim=0)

class AudioChunkFromURL(AudioChunk):
    def __init__(self, url:str, samplerate=8000, dtype=torch.float16, **kwargs):
        response = requests.get(url)
        sound_bytes = BytesIO(response.content)
        audio, samplerate_in = sf.read(sound_bytes)
        audio = torch.from_numpy(audio).to(dtype)
        super().__init__(audio, samplerate_in, **kwargs)
        if samplerate_in != samplerate:
            self.resample(samplerate)
