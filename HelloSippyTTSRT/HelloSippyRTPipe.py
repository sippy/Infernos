try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

import sys
from typing import Union, List, Any, Callable, Optional, Tuple
import contextlib, time
import random
from os.path import exists as path_exists
import weakref
from queue import Queue, Empty as QueueEmpty
import uuid

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

from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGanConfig, SpeechT5HifiGan, SpeechT5Processor, \
        SpeechT5Config, set_seed

class ErrMaxSessReached(Exception): pass

class SessCmd: pass

class SessSyncCmd(SessCmd):
    live: List[uuid.UUID]
    def __init__(self, sessions:weakref.WeakValueDictionary[InfernGlobals]): self.live = tuple(sorted(sessions.keys()))

class SessDispatchCmd(SessCmd):
    session: uuid.UUID
    def __init__(self, session_id:uuid.UUID): self.session = session_id

class HelloSippyRTPipe: pass

class HelloSippyPlayRequest(SessDispatchCmd):
    text:str
    speaker:torch.Tensor
    dispatch:weakref.ref[Queue]
    def __init__(self, session_id:uuid.UUID, text:str, speaker:torch.Tensor, dispatch:weakref.ref[Queue]): self.text, self.speaker, self.dispatch = (super().__init__(session_id), text, speaker, dispatch)[1:] 

class HelloSippyPipeState:
    session:uuid.UUID
    speaker_embeddings:Optional[torch.Tensor] = None
    encoder_last_hidden_state:Optional[torch.Tensor] = None
    output_sequence:Optional[torch.Tensor] = None
    encoder_attention_mask:Optional[torch.Tensor] = None
    pre_frames:Optional[torch.Tensor] = None
    ends_at:int = -1
    minlen:Optional[int] = None
    maxlen:Optional[int] = None
    dispatch:weakref.ref[Queue]

    def __init__(self, pp:HelloSippyRTPipe, req:HelloSippyPlayRequest):
        self.session, self.dispatch = req.session, req.dispatch
        inputs = pp.processor(text=req.text, return_tensors="pt")["input_ids"]
        self.speaker_embeddings = req.speaker.to(pp.model.device)
        self.encoder_attention_mask = torch.ones_like(inputs:=inputs.to(pp.model.device))
        encoder_out = pp.model.speecht5.encoder(
            input_values=inputs,
            attention_mask=self.encoder_attention_mask,
            return_dict=True,
        )
        self.encoder_last_hidden_state = encoder_out.last_hidden_state
        self.pre_frames = torch.zeros(1, pp.pre_nframes, pp.model.config.num_mel_bins, device=pp.model.device)
        self.maxlen = int(self.encoder_last_hidden_state.size(1) * pp.maxlenratio / pp.model.config.reduction_factor)
        self.minlen = int(self.encoder_last_hidden_state.size(1) * pp.minlenratio / pp.model.config.reduction_factor)

        # Start the output sequence with a mel spectrum that is all zeros.
        self.output_sequence = self.encoder_last_hidden_state.new_zeros(1, 1, pp.model.config.num_mel_bins)

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
    dispatch:List[weakref.ref[Queue]]
    res_queue:Optional[Queue] = None
    sessions:List[uuid.UUID]

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
        self.dispatch = [s.dispatch for s in states]

import threading

from elperiodic.ElPeriodic import ElPeriodic

from time import monotonic

class trp_thread(threading.Thread):
    queue: Queue
    queue_out: Optional[Queue] = None
    elp: Optional[ElPeriodic] = None
    period = None
    def __init__(self, period:float=0.0, noreturn:bool=False):
        self.queue = Queue()
        if not noreturn: self.queue_out = Queue()
        if period > 0.0: self.period = period #self.elp = ElPeriodic(1.0 / period)
        super().__init__(target=self.__thread)
        self.daemon = True
        self.start()

    def __call__(self, func):
        #raise Exception(f'__call__ {args=} {kwargs=}')
        def __call(*args, **kwargs):
            #raise Exception(f'__call {args=} {kwargs=}')
            t = monotonic()
            self.queue.put((func, args, kwargs))
            ex, res = self.queue_out.get()
            if ex: raise ex
            return res
        def __call_noret(*args, **kwargs):
            self.queue.put((func, args, kwargs))
        return __call if self.queue_out else __call_noret
        #return self.queue_out.get()

    def __thread(self):
        while True:
            a = self.queue.get()
            if a is None: break
            func, args, kwargs = a
            st = monotonic()
            try: res = (None, func(*args, **kwargs))
            except Exception as ex:res = (ex, None)
            et = monotonic()
            if self.queue_out: self.queue_out.put(res)
            elif res[0]: raise res[0]
            if self.elp: self.elp.procrastinate()
            if self.period and (et - st) < self.period: time.sleep(self.period - (et - st))

    def __del__(self):
        print('del')
        if not hasattr(self, 'queue'): return
        self.queue.put(None)
        self.join()
        self.func = None

import torchaudio.transforms as T

class InfernSession:
    _cmd_queue:Queue
    id:uuid.UUID
    default_speaker:torch.Tensor
    def __init__(self, queue, default_speaker:torch.Tensor): self.id, self._cmd_queue, self.default_speaker = uuid.uuid4(), queue, default_speaker
    def play(self, text:str, dispatch:Queue, speaker:Optional[torch.Tensor] = None):
        cmd = HelloSippyPlayRequest(self.id, text, speaker if speaker else self.default_speaker, weakref.ref(dispatch))
        self._cmd_queue.put(cmd)

class HelloSippyRTPipe:
    _main_thread_id: int
    _sync_queue: Queue
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
    sessions: weakref.WeakValueDictionary[InfernSession]
    max_sessions: int = 50

    def __init__(self, device, **kwa):
        self._main_thread_id = threading.get_ident()
        self._sync_queue = Queue()
        self.sessions = weakref.WeakValueDictionary()
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

    def alloc_session(self, speaker:Optional[torch.Tensor]=None) -> Tuple[InfernSession, HelloSippyPipeState]:
        assert threading.get_ident() == self._main_thread_id
        if len(self.sessions) >= self.max_sessions: raise ErrMaxSessReached(f'No more sessions available {self.max_sessions=}')
        if not speaker: speaker = self.get_rand_voice()
        rv = InfernSession(self._sync_queue, speaker)
        self.sessions[rv.id] = rv
        ss = SessSyncCmd(self.sessions)
        self._sync_queue.put(ss)
        return rv

    def savetensor(self, tensor:torch.Tensor, name:str):
        fname = f'{name}{self.saveidx}.npy'
        np.save(fname, tensor.cpu().numpy())
  
    @trp_thread(noreturn=True)
    def synchronize(self, state:Optional[HelloSippyPipeStateBatched], res_queue=None) -> None:
        ssq = []
        try:
            while True: ssq.append(self._sync_queue.get_nowait())
        except QueueEmpty: pass
        assert all(isinstance(x, SessCmd) for x in ssq)
        syncs, reqs = [x for x in ssq if isinstance(x, SessSyncCmd)], [x for x in ssq if not isinstance(x, SessSyncCmd)]
        if len(syncs) == 0 and (not state or len(reqs) == 0): return (self.main_gen(state), None)[-1]
        print(f'{len(syncs)=} {len(reqs)=} {syncs[-1]=}')
        live = syncs[-1].live
        if len(live) == 0: return (self.main_gen(None), None)[-1]
        reqs_live = [x for x in reqs if x.session in live]
        if len(reqs_live) == 0: return (self.main_gen(state), None)[-1]
        with self.cuda_lock:
            new_states = [HelloSippyPipeState(self, r) for r in reqs_live]
            if state: raise Exception("FIXME: NOT IMPLEMENTED")
            state = HelloSippyPipeStateBatched(new_states, self.model.device)
            state.res_queue = res_queue
        #raise Exception(f'{len(ssq)=} {reqs_live=} {live=} {len(self.sessions)=}')
        self.main_gen(state)

    @trp_thread(period=8*256*4/16000, noreturn=True)
    def main_gen(self, state:Optional[HelloSippyPipeStateBatched]) -> None:
        if not state: return (self.synchronize(None), None)[-1]
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

        self.unbatch_and_dispatch(state)

    @trp_thread(noreturn=True)
    def unbatch_and_dispatch(self, state: HelloSippyPipeStateBatched):
        with self.cuda_lock:
            audio, sr_rr = self.resampler(state.audio), self.model_sr / self.dispatch_sr
            end_idx = state.idx - 1
            eoff = (self.pre_nframes // self.model.config.reduction_factor)
            eidx = end_idx - eoff
            stepsize = int(256 * 2 / sr_rr)
            for i, cbq in [(i, cbq) for i, _cbq in enumerate(state.dispatch) if _cbq is not None and (cbq:=_cbq()) is not None]:
                ends_at_rel = (end_idx - (ends_at + eoff if (ends_at:=state.ends_at[i].item()) >= 0 and ends_at <= eidx else end_idx)) * stepsize
                if ends_at >= 0 and ends_at > eidx: print(f"gotcha {ends_at=} {eidx=}")
                cbq.put(audio[i][ends_at_rel:])
                if ends_at >= 0 and ends_at <= eidx:
                    cbq.put(None)
                    state.dispatch[i] = None
            mask = ((state.ends_at < 0) | (state.ends_at > eidx))
            if False and torch.sum(mask) < (state.ends_at.size(0) // 2):
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
            if torch.all(~mask):
                if state.res_queue: state.res_queue.put(state)
            else: self.synchronize(state)

    def get_rand_voice(self):
        s_index = torch.randint(0, len(self.speaker_embeddings), (1,)).item()
        rv = self.speaker_embeddings[s_index]
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
    from random import choices
    from utils.tts import smith_set, bender_set, hal_set
    n = 50
    prompts = choices([y for x in smith_set() + bender_set() + hal_set() for y in x.split('|')], k=n)
    #prompts = prompts
    #prompts = [prompts[0] for _ in range(n)]
    class res_cb(threading.Thread):
        def __init__(self, n, name='dispatch'):
            super().__init__(target=self.__thread)
            self.n, self.name = n, name
            if self.name == 'dispatch': self.data = torch.empty(0)
            self.q = Queue()
            self.daemon = True
            self.start()

        def __thread(self):
            while (y:=self.q.get()) is not None: self.data = torch.cat((self.data, y.cpu()))
            self.eos()

        def eos(self):
            sys.stdout.write(f'eos({self.n}) {self.data.shape=}\n')
            sys.stdout.flush()
            sf.write(f'out_{self.n}.wav', self.data.numpy(), 8000, 'PCM_16')

    params = {'hidden_dropout':0.0, 'positional_dropout':0.0, 'speech_decoder_prenet_dropout':0.0,
              'activation_dropout':0.0, 'encoder_layerdrop':0.0, 'decoder_layerdrop':0.0, 'attention_dropout':0.0,
              'speech_decoder_postnet_dropout':0.0, 'feat_proj_dropout':0.0}
    sp = HelloSippyRTPipe('xpu')
    if ipex is not None:
        sp.model = ipex.optimize(sp.model)
        sp.vocoder = ipex.optimize(sp.vocoder)
        sp.chunker = ipex.optimize(sp.chunker)

    s1 = [sp.alloc_session() for i in range(50)]
    del s1
    s2 = [((s:=sp.alloc_session()), (r:=res_cb(n)), s.play(p, r.q)) for n, p in enumerate(prompts)]
    res_queue = Queue()
    sp.synchronize(None, res_queue=res_queue)
    res = res_queue.get()
    from time import sleep
    sleep(1)
    return(1)

    def init_states(states):
        d_callbacks = [res_cb(n, 'dispatch', lambda x:x[0].shape) for n, _ in enumerate(states)]
        e_callbacks = [d.eos for d in d_callbacks]
        for state, d_cb, e_cb in zip(states, d_callbacks, e_callbacks): state.dispatch, state.eos_cb = d_cb, e_cb
        return states
    seed_RNGs()
    states = [sp.once(x) for x in prompts]
    init_states(states)
    states = sp.batch_for_main_gen(states)
    states.res_queue = Queue()
    sp.synchronize(states)
    with Timing("main_gen: "):
        state = states.res_queue.get()
    #while state.next is not None:
    #    state = state.next(state)
    #exit(1)
    with Timing("once: "):
        seed_RNGs()
        states = [sp.once(x) for x in prompts]
    init_states(states)
    #state1 = sp.once('Hello, world!')
    #state2 = sp.once('How are you doing today?')
    #state3 = sp.once('I am doing well, thank you very much.')
    with Timing("batch_for_main_gen: "):
        states = sp.batch_for_main_gen(states)
    states.res_queue = Queue()
    sp.synchronize(states)
    with Timing("main_gen: "):
        state = states.res_queue.get()

if __name__ == '__main__' and (r:=main()) not in (None, 0): raise RuntimeError(f'main() returned {r}')
