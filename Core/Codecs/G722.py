from G722 import G722
import torch
import numpy as np

from Core.AudioChunk import AudioChunk

class G722Codec():
    codec:G722
    default_sr:int = 8000
    default_br:int = 64000
    pt:int = 9 # G.722
    rm:str = 'G722/8000'
    _device:str = 'cpu'

    def __init__(self):
        self.codec = G722(self.default_sr, self.default_br)

    def encode(self, audio_tensor:torch.Tensor):
        # Scale from [-1, 1] to [-32768, 32767]
        audio_scaled = torch.clamp(audio_tensor * 32767.0, -32768, 32767).to(torch.int16).numpy()

        # Shift and look up in the conversion table
        audio_enc = self.codec.encode(audio_scaled)

        return audio_enc

    def decode(self, audio_enc:bytes, resample:bool=True, sample_rate:int=default_sr):
        # Use ulaw_to_pcm table to convert each µ-law value to PCM value
        audio_pcm = torch.tensor(self.codec.decode(audio_enc)).to(self._device)

        # Scale from [-32768, 32767] to [-1, 1]
        audio_float = audio_pcm.float() / 32767.0

        chunk = AudioChunk(audio_float, self.default_sr)
        if resample and sample_rate != self.default_sr:
            chunk.resample(sample_rate)
        return chunk

    def device(self): return self._device

    def to(self, device):
        self._device = device
        return self

    def silence(self, nframes:int):
        return self.encode(torch.zeros(self.e2d_frames(nframes), dtype=torch.int16))

    def e2d_frames(self, enframes:int, out_srate:int=default_sr):
        #assert out_srate % self.default_sr == 0
        return enframes * (1 if self.default_sr == 8000 else 2) * out_srate // self.default_sr

    def d2e_frames(self, dnframes:int, in_srate:int=default_sr):
        #assert in_srate % self.default_sr == 0
        return dnframes * self.default_sr // ((1 if self.default_sr == 8000 else 2) * in_srate)
