from typing import Tuple, Dict
import torch
import torchaudio.transforms as T
import audioop

from Core.AudioChunk import AudioChunk

_pcm_to_ulaw_ct = torch.zeros(65536, dtype=torch.uint8)
for i in range(-32768, 32768):
    pcm_data = i.to_bytes(2, 'little', signed=True)
    ulaw_data = audioop.lin2ulaw(pcm_data, 2)
    ulaw_value = ulaw_data[0]  # Get the byte value from bytes
    _pcm_to_ulaw_ct[i + 32768] = ulaw_value  # Shift index to make it non-negative
_ulaw_to_pcm_ct = torch.zeros(256, dtype=torch.int16)
for i in range(256):
    # Convert each µ-law value back to PCM value
    ulaw_byte = i.to_bytes(1, 'little')
    pcm_data = audioop.ulaw2lin(ulaw_byte, 2)  # Convert µ-law byte to linear PCM
    pcm_value = int.from_bytes(pcm_data, 'little', signed=True)
    _ulaw_to_pcm_ct[i] = pcm_value

class G711Codec():
    default_sr:int = 8000
    pt:int = 0 # G.711u
    resamplers: Dict[Tuple[int, int], T.Resample]
    def __init__(self):
        self.resamplers = {}

    def encode(self, audio_tensor:torch.Tensor):
        # Scale from [-1, 1] to [-32768, 32767]
        audio_scaled = torch.clamp(audio_tensor * 32767.0, -32768, 32767).to(torch.int16)

        # Shift and look up in the conversion table
        audio_ulaw = _pcm_to_ulaw_ct[(audio_scaled + 32768).long()]

        return audio_ulaw

    def decode(self, ulaw_bytes:bytes, resample:bool=True, sample_rate:int=default_sr):
        # Convert byte string to a tensor of uint8
        ulaw_tensor = torch.tensor(list(ulaw_bytes), dtype=torch.uint8)

        # Use ulaw_to_pcm table to convert each µ-law value to PCM value
        audio_pcm = _ulaw_to_pcm_ct[ulaw_tensor.long()]

        # Scale from [-32768, 32767] to [-1, 1]
        audio_float = audio_pcm.float() / 32767.0

        if resample and sample_rate != self.default_sr:
            resampler = self.get_resampler(self.default_sr, sample_rate)
            return AudioChunk(resampler(audio_float), sample_rate)
        return AudioChunk(audio_float, self.default_sr)

    def get_resampler(self, from_sr:int, to_sr:int):
        key = (from_sr, to_sr)
        if (resampler:=self.resamplers.get(key, None)) is None:
            resampler = T.Resample(orig_freq=from_sr, new_freq=to_sr)
            self.resamplers[key] = resampler
        return resampler

    def device(self):
        global _pcm_to_ulaw_ct, _ulaw_to_pcm_ct
        assert _pcm_to_ulaw_ct.device == _ulaw_to_pcm_ct.device
        return _pcm_to_ulaw_ct.device

    def to(self, device):
        global _pcm_to_ulaw_ct, _ulaw_to_pcm_ct
        assert _pcm_to_ulaw_ct.device == _ulaw_to_pcm_ct.device
        _pcm_to_ulaw_ct = _pcm_to_ulaw_ct.to(device)
        _ulaw_to_pcm_ct = _ulaw_to_pcm_ct.to(device)
        self.resamplers = dict((k, v.to(device)) for k, v in self.resamplers.items())
        return self
