try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from typing import Tuple, List, Optional

import torch

from Cluster.InfernBatchedWorker import InfernBatchedWorker
from Core.AudioChunk import AudioChunk
from Core.VAD.SileroVADUtils import VADIteratorB, VADChannelState, VADBatchFromList

class VADChannel():
    audio_chunk_in: callable
    vad_chunk_in: callable
    decode: callable
    vad_buffer: bytes = b''
    state: VADChannelState
    active_start: Optional[int] = None
    active_buffer: torch.Tensor
    def __init__(self, audio_chunk_in:callable, vad_chunk_in: callable, decode: callable, device:str):
        self.audio_chunk_in = audio_chunk_in
        self.vad_chunk_in = vad_chunk_in
        self.decode = decode
        self.state = VADChannelState(device)
        self.active_buffer = torch.zeros(0).to('cpu')

    def ingest(self, svad:'SileroVADWorker', data: bytes, decode:callable):
        self.vad_buffer += data
        if len(self.vad_buffer) < svad.window_size_samples:
            return None
        chunk = decode(self.vad_buffer[:svad.window_size_samples])
        self.vad_buffer = self.vad_buffer[svad.window_size_samples:]
        svad.infer((self, chunk))
        #self.vad_chunk_in(chunk, True)

class SileroVADWorker(InfernBatchedWorker):
    max_batch_size: int = 200
    input_sr: int
    max_vad_frames: int
    def __init__(self, device, input_sr: int = 8000):
        super().__init__()
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad', force_reload=True)
        self.device = device
        self.model = model = model.eval().to(device)
        #for n, t in [(_t, getattr(_m, _t, None))
        #     for _m in (model._model_8k.decoder.rnn, model._model.decoder.rnn)
        #     for _t in dir(_m)
        #     if _t != 'graph']:
        #    if not isinstance(t, torch.Tensor): continue
        #    print(f'{n=} {t.is_contiguous()=}')

        self.vad_iterator = VADIteratorB(model, sampling_rate=input_sr)
        self.window_size_samples = 768 # number of samples in a single audio chunk
        self.input_sr = input_sr
        self.max_vad_frames = input_sr * 30 # 30 seconds for Whisper

    @torch.no_grad()
    def process_batch(self, wis:List[Tuple[VADChannel, torch.Tensor]]):
        from time import sleep
        #sleep(0.5)
        #print(f'InfernSTTWorker.process_batch: got {len(wis)=}')
        while len(wis) > 0:
            nbatch = []
            cbatch: List[VADChannel] = []
            pbatch: List[AudioChunk] = []
            sbatch: List[VADChannelState] = []
            for wi in wis:
                if (ch:=wi[0]) not in cbatch:
                    cbatch.append(ch)
                    pbatch.append(wi[1])
                    sbatch.append(ch.state)
                else:
                    nbatch.append(wi)
            wis = nbatch
            bstate = VADBatchFromList(sbatch)
            chunks = torch.stack([p.audio for p in pbatch], dim=0).to(self.device)
            self.vad_iterator(chunks, bstate=bstate, return_seconds=False)
            for i, (vc, p) in enumerate(zip(cbatch, pbatch)):
                sd = vc.state
                if sd.speech: print(f'speech_dict[{i}]={sd.speech} {sd.current_sample=}', end=' ')
                vc.active_buffer = torch.cat((vc.active_buffer, p.audio.cpu()))
                if sd.speech and 'start' in sd.speech:
                    assert vc.active_start is None, f'{vc.active_start=}'
                    vc.active_start = sd.speech['start']
                    poff = sd.current_sample - vc.active_start
                    assert poff > 0 and poff < vc.active_buffer.size(0), f'{poff=} {vc.active_buffer.size(0)=} {sd.current_sample=} {vc.active_start=}'
                    vc.active_buffer = vc.active_buffer[-poff:]
                elif sd.speech and 'end' in sd.speech:
                    active_end = sd.speech["end"]
                    assert vc.active_start is not None and active_end > vc.active_start, f'{vc.active_start=} {sd.temp_end=} {active_end=}'
                    assert sd.current_sample > active_end, f'{sd.current_sample=} {active_end=}'
                    poff = sd.current_sample - active_end
                    assert poff > 0 and poff < vc.active_buffer.size(0), f'{poff=} {vc.active_buffer.size(0)=} {sd.current_sample=} {active_end=}'
                    obuf = vc.active_buffer[:-poff]
                    assert obuf.size(0) == active_end - vc.active_start, f'{obuf.size(0)=} {vc.active_start=} {active_end=}'
                    vc.active_start = None
                    vc.vad_chunk_in(AudioChunk(obuf, self.input_sr))
                if vc.active_start is None:
                    vc.active_buffer = vc.active_buffer[:self.window_size_samples*2]
                elif vc.active_buffer.size(0) > self.max_vad_frames:
                    chunk = AudioChunk(vc.active_buffer[:self.max_vad_frames], self.input_sr)
                    vc.active_buffer = vc.active_buffer[self.max_vad_frames:]
                    vc.active_start += self.max_vad_frames
                    if sd.temp_end != 0:
                        print(f'{sd.current_sample=}: {sd.temp_end=} -> {vc.active_start=}')
                        if sd.temp_end < vc.active_start:
                            sd.temp_end = vc.active_start
                    vc.vad_chunk_in(chunk)
                vc.audio_chunk_in(p, vc.active_start is not None)
