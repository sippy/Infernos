import torch
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
    rm:str = 'PCMU/8000'

    def encode(self, audio_tensor:torch.Tensor):
        # Scale from [-1, 1] to [-32768, 32767]
        audio_scaled = torch.clamp(audio_tensor * 32767.0, -32768, 32767).to(torch.int16)

        # Shift and look up in the conversion table
        audio_ulaw = _pcm_to_ulaw_ct[(audio_scaled + 32768).long()]

        return audio_ulaw.cpu().numpy().tobytes()

    def decode(self, ulaw_bytes:bytes, resample:bool=True, sample_rate:int=default_sr):
        # Convert byte string to a tensor of uint8
        ulaw_tensor = torch.tensor(list(ulaw_bytes), dtype=torch.uint8)

        # Use ulaw_to_pcm table to convert each µ-law value to PCM value
        audio_pcm = _ulaw_to_pcm_ct[ulaw_tensor.long()]

        # Scale from [-32768, 32767] to [-1, 1]
        audio_float = audio_pcm.float() / 32767.0

        chunk = AudioChunk(audio_float, self.default_sr)
        if resample and sample_rate != self.default_sr:
            chunk.resample(sample_rate)
        return chunk

    def device(self):
        global _pcm_to_ulaw_ct, _ulaw_to_pcm_ct
        assert _pcm_to_ulaw_ct.device == _ulaw_to_pcm_ct.device
        return _pcm_to_ulaw_ct.device

    def to(self, device):
        global _pcm_to_ulaw_ct, _ulaw_to_pcm_ct
        assert _pcm_to_ulaw_ct.device == _ulaw_to_pcm_ct.device
        _pcm_to_ulaw_ct = _pcm_to_ulaw_ct.to(device)
        _ulaw_to_pcm_ct = _ulaw_to_pcm_ct.to(device)
        return self

    def e2d_frames(self, enframes:int, out_srate:int=default_sr):
        assert out_srate % self.default_sr == 0
        return enframes * out_srate // self.default_sr

    def d2e_frames(self, dnframes:int, in_srate:int=default_sr):
        assert in_srate % self.default_sr == 0
        return dnframes * self.default_sr // in_srate

    def silence(self, nframes:int):
        return b'\xff' * nframes
