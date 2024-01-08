try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from typing import Union, List, Any, Callable, Optional
import contextlib, time
import random
from os.path import exists as path_exists
import weakref

import numpy as np

import torch
from torch.nn import functional as F
from datasets import load_dataset

try: from config.InfernGlobals import InfernGlobals
except ModuleNotFoundError:
    from sys import path as sys_path
    from os import getcwd
    sys_path.append(getcwd())
    from config.InfernGlobals import InfernGlobals

from HelloSippyTTSRT.HelloSippyRT import AmendmentNetwork1, AmendmentNetwork1Config

def pad_tensor_to_target(tensor, target_size):
    if tuple(tensor.shape) == tuple(target_size): return tensor
    # Get the current size of the tensor
    current_size = tensor.size()
    # Calculate the padding needed for each dimension
    padding = []
    for c, t in zip(current_size[::-1], target_size[::-1]):  # Reverse to start from the last dimension
        padding.extend([0, max(t - c, 0)])
    # Apply padding
    return F.pad(tensor, tuple(padding), "constant", 0.0)

class HelloSippyPipeState:
    speaker_embeddings:Union[torch.Tensor,None] = None
    encoder_last_hidden_state:Union[torch.Tensor,None] = None
    output_sequence:Union[torch.Tensor,None] = None
    encoder_attention_mask:Union[torch.Tensor,None] = None
    pre_frames:Union[torch.Tensor,None] = None
    ends_at:int = -1
    minlen:Union[int,None] = None
    maxlen:Union[int,None] = None
    idx: int = 0
    dispatch:Union[Callable,None] = None
    eos_cb:Union[Callable,None] = None
    next:Any = None

class HelloSippyPipeStateBatched:
    speaker_embeddings:torch.Tensor
    encoder_last_hidden_state:torch.Tensor
    output_sequence:torch.Tensor = None
    past_key_values:List[Union[torch.Tensor, None]] = None
    encoder_attention_mask:Union[torch.Tensor,None] = None
    pre_frames:Union[torch.Tensor,None] = None
    ends_at:Union[torch.Tensor,None] = None
    minlen:[torch.Tensor,None] = None
    maxlen:[torch.Tensor,None] = None
    idx: int = 0
    dispatch:List[weakref.ref[Callable[[], Optional[None]]]]
    eos_cb:List[weakref.ref[Callable[[], Optional[None]]]]
    next:Any = None

    def __init__(self, states: List[HelloSippyPipeState], device):
        self.speaker_embeddings = torch.cat([s.speaker_embeddings for s in states], dim=0)
        encoder_last_hidden_states = [s.encoder_last_hidden_state for s in states]
        encoder_attention_masks = [s.encoder_attention_mask for s in states]
        max_d1 = max(mask.size(1) for mask in encoder_attention_masks)
        padded_states = [pad_tensor_to_target(state, (state.size(0), max_d1, state.size(2))) for state in encoder_last_hidden_states]
        padded_masks = [torch.nn.functional.pad(mask, (0, max_d1 - mask.size(1)), 'constant', 0) for mask in encoder_attention_masks]
        self.encoder_last_hidden_state = torch.cat(padded_states, dim=0)
        self.encoder_attention_mask = torch.cat(padded_masks, dim=0)
        self.pre_frames = torch.cat([s.pre_frames for s in states], dim=0)
        self.maxlen = torch.tensor([s.maxlen for s in states], dtype=torch.long, device=device)
        self.minlen = torch.tensor([s.minlen for s in states], dtype=torch.long, device=device)
        self.output_sequence = torch.cat([s.output_sequence for s in states], dim=0)
        self.past_key_values = None
        self.ends_at = torch.tensor([s.ends_at for s in states], dtype=torch.long, device=device)
        self.idx = 0
        self.dispatch = [None if (d:=s.dispatch) is None else weakref.ref(d) for s in states]
        self.eos_cb = [None if (d:=s.eos_cb) is None else weakref.ref(d) for s in states]

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGanConfig, SpeechT5HifiGan, SpeechT5Processor, \
        SpeechT5Config, set_seed

import threading
from queue import Queue

from elperiodic.ElPeriodic import ElPeriodic

class trp_thread(threading.Thread):
    queue: Queue
    queue_out: Queue
    period = None
    def __init__(self, period:float):
        self.queue = Queue()
        self.queue_out = Queue()
        self.elp = ElPeriodic(1.0 / period)
        super().__init__(target=self.__thread)
        self.daemon = True
        self.start()

    def __call__(self, func):
        #raise Exception(f'__call__ {args=} {kwargs=}')
        def __call(*args, **kwargs):
            #raise Exception(f'__call {args=} {kwargs=}')
            self.queue.put((func, args, kwargs))
            ex, res = self.queue_out.get()
            if ex: raise ex
            return res
        return __call
        #return self.queue_out.get()

    def __thread(self):
        while True:
            a = self.queue.get()
            if a is None: break
            func, args, kwargs = a
            try: res = (None, func(*args, **kwargs))
            except Exception as ex: res = (ex, None)
            self.queue_out.put(res)
            self.elp.procrastinate()

    def __del__(self):
        if not hasattr(self, 'queue'): return
        self.queue.put(None)
        self.join()
        self.func = None

import torchaudio.transforms as T

class HelloSippyRTPipe:
    processor: SpeechT5Processor
    model: SpeechT5ForTextToSpeech
    chunker: AmendmentNetwork1
    resampler: T.Resample
    minlenratio: float = 0.0
    maxlenratio: float = 20.0
    threshold: float = 0.5
    chunk_size: int = 8
    pre_nframes: int = 2
    post_nframes: int = 2
    model_sr: int = 16000
    dispatch_sr: int = 8000

    def __init__(self, device, **kwa):
        self.cuda_lock = InfernGlobals().torcher
        with self.cuda_lock:
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            mc = SpeechT5Config(max_speech_positions=4000, **kwa)
            model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts",
                                                            config=mc).to(device)
            model.eval()
            self.model = model
            _vc_conf = SpeechT5HifiGanConfig()
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",
                                                    config = _vc_conf).to(device)
            vocoder.eval()
            self.vocoder = vocoder
            self.c_conf = AmendmentNetwork1Config()
            chunker = AmendmentNetwork1.from_pretrained("sobomax/speecht5-rt.post_vocoder.v2",
                                                        config=self.c_conf)
            chunker = chunker.to(device)
            chunker.eval()
            self.chunker = chunker
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = [torch.tensor(ed["xvector"], device='cpu').unsqueeze(0)
                                        for ed in sorted(embeddings_dataset, key=lambda x: x['filename'])]
            for x in [_x for x in (self.model.parameters, self.vocoder.parameters, self.chunker.parameters) for _x in x()] + self.speaker_embeddings: x.requires_grad = False
            self.resampler = T.Resample(orig_freq=self.model_sr, new_freq=self.dispatch_sr).to(device)

    def savetensor(self, tensor:torch.Tensor, name:str):
        fname = f'{name}{self.saveidx}.npy'
        np.save(fname, tensor.cpu().numpy())

    def once(self, text:str, speaker=None) -> HelloSippyPipeState:
        state = HelloSippyPipeState()
        inputs = self.processor(text=text, return_tensors="pt")["input_ids"]
        #for self.saveidx in range(100):
        #    fname = f'inputs{self.saveidx}.npy'
        #    if path_exists(fname): continue
        #    np.save(fname, inputs.cpu().numpy())
        #    break
        #else: raise RuntimeError(f'Could not save inputs{self.saveidx}.npy')

        with self.cuda_lock:
            if speaker is None:
                speaker = self.get_rand_voice()
            state.speaker_embeddings = speaker
            state.encoder_attention_mask = torch.ones_like(inputs:=inputs.to(self.model.device))
            encoder_out = self.model.speecht5.encoder(
                input_values=inputs,
                attention_mask=state.encoder_attention_mask,
                return_dict=True,
            )
            state.encoder_last_hidden_state = encoder_out.last_hidden_state
            state.pre_frames = torch.zeros(1, self.pre_nframes, self.model.config.num_mel_bins, device=self.model.device)
            state.maxlen = int(state.encoder_last_hidden_state.size(1) * self.maxlenratio / self.model.config.reduction_factor)
            state.minlen = int(state.encoder_last_hidden_state.size(1) * self.minlenratio / self.model.config.reduction_factor)

            # Start the output sequence with a mel spectrum that is all zeros.
            state.output_sequence = state.encoder_last_hidden_state.new_zeros(1, 1, self.model.config.num_mel_bins)
        state.next = self.batch_for_main_gen
        return state
    
    def batch_for_main_gen(self, states: List[HelloSippyPipeState]) -> HelloSippyPipeStateBatched:
        with self.cuda_lock:
           state = HelloSippyPipeStateBatched(states, self.model.device)

        state.next = self.main_gen
        return state

    @trp_thread(period=8*256*4/16000)
    def main_gen(self, state: HelloSippyPipeStateBatched) -> HelloSippyPipeStateBatched:
        with self.cuda_lock:
            batch_size = state.output_sequence.size(0)
            spectrogram = torch.zeros(batch_size, 0, self.model.config.num_mel_bins, device=self.model.device)
            output_len = (self.chunk_size * 4) + (self.post_nframes if state.idx == 0 else 0)
            while spectrogram.size(1) < output_len:
                decoder_hidden_states = self.model.speecht5.decoder.prenet(state.output_sequence, state.speaker_embeddings)[:, -1:]
                decoder_out = self.model.speecht5.decoder.wrapped_decoder(
                    hidden_states=decoder_hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=state.encoder_last_hidden_state,
                    encoder_attention_mask=state.encoder_attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    output_attentions=False,
                    return_dict=True,
                )
                last_decoder_output = decoder_out.last_hidden_state[:, -1, :]
                state.past_key_values = decoder_out.past_key_values

                # Predict the new mel spectrum for this step in the sequence.
                spectrum = self.model.speech_decoder_postnet.feat_out(last_decoder_output)

                spectrum = spectrum.view(batch_size, self.model.config.reduction_factor, self.model.config.num_mel_bins)
                spectrogram = torch.cat((spectrogram, spectrum), dim=1)

                # Extend the output sequence with the new mel spectrum.
                spv = spectrum[:, -1:, :] #.view(spectrum.size(0), 1, self.model.config.num_mel_bins)
                state.output_sequence = torch.cat((state.output_sequence, spv), dim=1)

                # Predict the probability that this is the stop token.
                prob = self.model.speech_decoder_postnet.prob_out(last_decoder_output).sigmoid()

                # Finished when stop token or maximum length is reached.
                # if state.idx >= state.minlen and (int(sum(prob >= self.threshold)) > 0 or state.idx >= state.maxlen):
                #print(f"{(state.minlen <= state.idx)=} {(torch.sum(prob >= self.threshold, (1,)) > 0)=}")
                #raise Exception(f"{(state.maxlen <= state.idx)=}")
                state.ends_at = torch.where((state.ends_at < 0) & (state.minlen <= state.idx) & ((torch.sum(prob >= self.threshold, (1,)) > 0) | (state.maxlen <= state.idx)),
                                             state.idx, state.ends_at)
                state.idx += 1
            spectrogram = self.model.speech_decoder_postnet.postnet(spectrogram)
            ##range_tensor = torch.arange(spectrogram.shape[1], device=state.ends_at.device).unsqueeze(0).expand(batch_size, -1) + (state.idx - 1)
            ##ends=state.ends_at.unsqueeze(1)
            ##print(f'{range_tensor.shape=} {torch.nonzero(~ends)=}')
            ##mask = ((range_tensor <= ends) | (ends < 0)).unsqueeze(-1)
            ##spectrogram = torch.where(mask, spectrogram, 0.0)
            #raise Exception(f'{spectrogram.shape=} {state.ends_at.shape=} {mask=}')
            spectrogram = torch.cat((state.pre_frames, spectrogram), dim=1)
            eframes = self.pre_nframes + self.post_nframes
            state.pre_frames = spectrogram[:, -eframes:, :]
            nchunks = spectrogram.size(1) // self.chunk_size
            spectrogram = torch.cat([spectrogram[:, i*self.chunk_size:(i+1)*self.chunk_size+eframes, :] for i in range(nchunks)], dim=0)
            audio = self.vocoder(spectrogram)
            audio = self.chunker(spectrogram, audio)
            slices = audio.split(batch_size, dim=0)
            state.audio = torch.cat(slices, dim=1)
            #self.savetensor(state.audio, f'audio_{state.idx}_')
        print(f'{state.ends_at.cpu().numpy()=} {state.audio.shape=}')

        state.next = self.unbatch_and_dispatch
        return state

    def unbatch_and_dispatch(self, state: HelloSippyPipeStateBatched) -> HelloSippyPipeStateBatched:
        audio, sr_rr = self.resampler(state.audio), self.model_sr / self.dispatch_sr
        end_idx = state.idx - 1
        eoff = (self.pre_nframes // self.model.config.reduction_factor)
        eidx = end_idx - eoff
        stepsize = int(256 * 2 / sr_rr)
        for i, cb in [(i, cb) for i, _cb in enumerate(state.dispatch) if _cb is not None and (cb:=_cb()) is not None]:
            ends_at_rel = (end_idx - (ends_at + eoff if (ends_at:=state.ends_at[i].item()) >= 0 and ends_at <= eidx else end_idx)) * stepsize
            if ends_at >= 0 and ends_at > eidx: print(f"gotcha {ends_at=} {eidx=}")
            cb(audio[i][ends_at_rel:])
            if ends_at >= 0 and ends_at <= eidx:
                state.dispatch[i] = None
                if (eos_cb:=state.eos_cb[i]()) is not None: eos_cb()
                state.eos_cb[i] = None
        mask = ((state.ends_at < 0) | (state.ends_at > eidx))
        if torch.sum(mask) < (state.ends_at.size(0) // 2):
            for tn in ('speaker_embeddings', 'encoder_last_hidden_state', 'output_sequence',
                      'encoder_attention_mask', 'pre_frames', 'ends_at', 'minlen', 'maxlen'):
                t = getattr(state, tn)
                t = t[mask].contiguous()
                setattr(state, tn, t)
            past_key_values = [list(x) for x in state.past_key_values]
            for past_key_value, idx, t in [(x, i, _x) for x in past_key_values for i, _x in enumerate(x)]:
                assert id(past_key_value[idx]) == id(t)
                past_key_value[idx] = t[mask].contiguous()
            state.past_key_values = tuple([tuple(x) for x in past_key_values])
            goodpos = torch.nonzero(mask).squeeze(1).tolist()
            state.dispatch = [x for i, x in enumerate(state.dispatch) if i in goodpos]
            state.eos_cb = [x for i, x in enumerate(state.eos_cb) if i in goodpos]
        state.next = None if torch.all(~mask) else self.main_gen
        return state

    def get_rand_voice(self):
        s_index = torch.randint(0, len(self.speaker_embeddings), (1,)).item()
        rv = self.speaker_embeddings[s_index].to(self.model.device)
        return rv

class Timing(contextlib.ContextDecorator):
  def __init__(self, prefix="", on_exit=None, enabled=True): self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled
  def __enter__(self): self.st = time.perf_counter_ns()
  def __exit__(self, *exc):
      self.et = time.perf_counter_ns() - self.st
      if self.enabled: print(f"{self.prefix}{self.et*1e-6:6.2f} ms"+(self.on_exit(self.et) if self.on_exit else ""))

def seed_RNGs():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def main():
    import soundfile as sf
    seed_RNGs()
    from utils.tts import smith_set
    n = 50
    prompts = [y for x in smith_set() for y in x.split('|')][:n]
    #prompts = [prompts[0] for _ in range(n)]
    class res_cb:
        def __init__(self, n, name, f):
            self.n, self.name, self.f = n, name, f
            if self.name == 'dispatch': self.data = torch.empty(0)
        def __call__(self, *x):
            if self.name == 'dispatch': self.data = torch.cat((self.data, x[0].cpu())); print(f'{self.name} {self.data.shape=}')
            print(f'{self.name}({self.n}) {self.f(x)=}')

        def eos(self):
            print(f'eos({self.n}) {self.data.shape=}')
            sf.write(f'out_{self.n}.wav', self.data.numpy(), 8000, 'PCM_16')

    d_callbacks = [res_cb(_n, 'dispatch', lambda x:x[0].shape) for _n in range(n)]
    e_callbacks = [d.eos for d in d_callbacks]
    params = {'hidden_dropout':0.0, 'positional_dropout':0.0, 'speech_decoder_prenet_dropout':0.0,
              'activation_dropout':0.0, 'encoder_layerdrop':0.0, 'decoder_layerdrop':0.0, 'attention_dropout':0.0,
              'speech_decoder_postnet_dropout':0.0, 'feat_proj_dropout':0.0}
    sp = HelloSippyRTPipe('xpu')
    if ipex is not None:
        sp.model = ipex.optimize(sp.model)
        sp.vocoder = ipex.optimize(sp.vocoder)
        sp.chunker = ipex.optimize(sp.chunker)

    seed_RNGs()
    states = [sp.once(x) for x in prompts]
    for state, d_cb, e_cb in zip(states, d_callbacks, e_callbacks): state.dispatch, state.eos_cb = d_cb, e_cb
    state = sp.batch_for_main_gen(states)
    while state.next is not None:
        state = state.next(state)
    exit(1)
    with Timing("once: "):
        seed_RNGs()
        states = [sp.once(x) for x in prompts]
    #state1 = sp.once('Hello, world!')
    #state2 = sp.once('How are you doing today?')
    #state3 = sp.once('I am doing well, thank you very much.')
    with Timing("batch_for_main_gen: "):
        state = sp.batch_for_main_gen(states)
    nruns = 0
    with Timing("main_gen: "):
        while state.next is not None:
            state = state.next(state)
            nruns += 1
    print(f'{nruns=}')

if __name__ == '__main__' and (r:=main()) not in (None, 0): raise RuntimeError(f'main() returned {r}')
