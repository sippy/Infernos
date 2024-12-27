from typing import List, Dict, Optional
import torch

class VADChannelState:
    triggered: bool = False
    temp_end: int = 0
    current_sample: int = 0
    speech: Optional[Dict[str, int]] = None
    model_state: List[torch.Tensor]
    def __init__(self, device:str):
        self.model_state = [torch.zeros(2, 64).to(device), torch.zeros(2, 64).to(device)]

class VADBatchState:
    batch_size: int
    channels: List[VADChannelState]
    device: str
    def __init__(self, batch_size, device:str='cpu'):
        self.batch_size = batch_size
        self.channels = [VADChannelState(device) for _ in range(batch_size)]

    def get_model_state(self):
        return [torch.stack([s.model_state[r] for s in self.channels], dim=1) for r in range(2)]

    def save_model_state(self, state:List[torch.Tensor]):
        for c, s1, s2 in zip(self.channels, state[0].unbind(1), state[1].unbind(1)):
            c.model_state = [s1, s2]

class VADBatchFromList(VADBatchState):
    def __init__(self, states:List[VADChannelState]):
        self.batch_size = len(states)
        self.channels = states

class VADIteratorB:
    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30,
                 ):

        """
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator does not support sampling rates other than [8000, 16000]')

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.model.reset_states()

    def __call__(self, x:torch.Tensor, bstate:Optional[VADBatchState]=None, return_seconds=False):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """

        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        if x.dim() == 1: x = x.unsqueeze(0)
        else: assert x.dim() == 2, f"Audio should be 1D or 2D tensor, but got {x.dim()}"

        batch_size = x.size(0)

        if bstate is None:
            bstate = VADBatchState(batch_size, device=x.device)
            self.model.reset_states()
        else:
            assert bstate.batch_size == batch_size, f"Batch size should be {batch_size}, but got {bstate.batch_size}"
            (_mc:=self.model._c)._h, _mc._c, _mc._last_sr, _mc._last_batch_size = bstate.get_model_state() + [self.sampling_rate, batch_size]

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)

        speech_probs = (y for y in self.model(x, self.sampling_rate).tolist())

        for speech_prob, channel in zip(speech_probs, bstate.channels):
            channel.current_sample += window_size_samples
            if (speech_prob >= self.threshold) and channel.temp_end:
                channel.temp_end = 0

            if (speech_prob >= self.threshold) and not channel.triggered:
                channel.triggered = True
                speech_pad_samples = self.speech_pad_samples if channel.current_sample > window_size_samples else 0
                speech_start = channel.current_sample - speech_pad_samples - window_size_samples
                channel.speech = {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}
                continue

            if (speech_prob < self.threshold - 0.15) and channel.triggered:
                if not channel.temp_end:
                    channel.temp_end = channel.current_sample
                if channel.current_sample - channel.temp_end < self.min_silence_samples:
                    channel.speech = None
                    continue
                else:
                    speech_end = channel.temp_end + self.speech_pad_samples - window_size_samples
                    channel.temp_end = 0
                    channel.triggered = False
                    channel.speech = {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}
                    continue

            channel.speech = None
        bstate.save_model_state([(_mc:=self.model._c)._h, _mc._c])
        #print(f'{bstate.model_state[0].size()=} {bstate.model_state[1].size()=}')
        return bstate
