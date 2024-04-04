import torch

class AudioChunk():
    samplerate: int
    audio:torch.Tensor
    track_id: int = 0
    def __init__(self, audio:torch.Tensor, samplerate:int):
        assert isinstance(audio, torch.Tensor)
        self.audio = audio
        self.samplerate = samplerate
