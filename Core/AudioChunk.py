import torch
import requests
import soundfile as sf
from io import BytesIO

from config.InfernGlobals import InfernGlobals as IG

class AudioChunk():
    samplerate: int
    audio:torch.Tensor
    track_id: int = 0
    def __init__(self, audio:torch.Tensor, samplerate:int):
        assert isinstance(audio, torch.Tensor)
        self.audio = audio
        self.samplerate = samplerate

class AudioChunkFromURL(AudioChunk):
    def __init__(self, url:str, samplerate=8000, dtype=torch.float16, **kwargs):
        response = requests.get(url)
        sound_bytes = BytesIO(response.content)
        audio, samplerate_in = sf.read(sound_bytes)
        audio = torch.from_numpy(audio)
        if samplerate_in != samplerate:
            audio = IG.get_resampler(samplerate_in, samplerate)(audio.to(torch.float))
        super().__init__(audio.to(dtype), samplerate, **kwargs)
