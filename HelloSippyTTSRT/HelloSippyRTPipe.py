try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

import sys, random, weakref, uuid
from typing import Union, List, Any, Callable, Optional, Tuple
import contextlib, time
from os.path import exists as path_exists
from queue import Queue, Empty as QueueEmpty
from dataclasses import dataclass

import numpy as np

import torch
from torch.nn import functional as F
from datasets import load_dataset
from methodtools import lru_cache

try: from config.InfernGlobals import InfernGlobals
except ModuleNotFoundError:
    from sys import path as sys_path
    from os import getcwd
    sys_path.append(getcwd())
    from config.InfernGlobals import InfernGlobals

from HelloSippyTTSRT.HelloSippyRT import AmendmentNetwork1, AmendmentNetwork1Config

def pad_tensor_to_target(tensor, target_size, pre=False):
    if tuple(tensor.shape) == tuple(target_size): return tensor
    # Get the current size of the tensor
    current_size = tensor.size()
    # Calculate the padding needed for each dimension
    padding = []
    for c, t in zip(current_size[::-1], target_size[::-1]):  # Reverse to start from the last dimension
        pad = max(t - c, 0)
        padding.extend([pad, 0] if pre else [0, pad])
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
    dispatch:callable
    def __init__(self, session_id:uuid.UUID, text:str, speaker:torch.Tensor, dispatch:callable):
        self.text, self.speaker, self.dispatch = text, speaker, dispatch
        super().__init__(session_id)

def make_tensor(x, pp): return torch.tensor([x], dtype=torch.long, device=pp.model.device)

def maybe_half(x): return x.to(memory_format=torch.channels_last, dtype=torch.bfloat16) if not isinstance(x, torch.Tensor) or len(x.shape) > 3 else x.to(dtype=torch.bfloat16)

class HelloSippyPipeState:
    session:uuid.UUID
    dispatch:callable
    inputs:torch.Tensor
    speaker_embeddings:torch.Tensor
    encoder_last_hidden_state:torch.Tensor
    output_sequence:torch.Tensor
    encoder_attention_mask:torch.Tensor
    pre_frames:torch.Tensor
    starts_at:torch.Tensor
    ends_at:torch.Tensor

    def __init__(self, pp:HelloSippyRTPipe, req:HelloSippyPlayRequest):
        self.session, self.dispatch = req.session, req.dispatch
        self.inputs = pp.processor(text=req.text, return_tensors="pt")["input_ids"].to(pp.model.device)
        self.speaker_embeddings = maybe_half(req.speaker).to(pp.model.device)
        self.encoder_attention_mask = torch.ones_like(self.inputs, dtype=torch.int)
        self.pre_frames = maybe_half(torch.zeros(1, pp.pre_nframes + pp.post_nframes, pp.model.config.num_mel_bins)).to(pp.model.device)
        self.starts_at = make_tensor(pp.post_nframes // pp.model.config.reduction_factor, pp)
        self.ends_at = make_tensor(-1, pp)

class HelloSippyPipeStateBatched:
    speaker_embeddings:torch.Tensor
    encoder_last_hidden_state:torch.Tensor
    output_sequence:torch.Tensor
    past_key_values:Optional[List[torch.Tensor]] = None
    encoder_attention_mask:torch.Tensor
    pre_frames:torch.Tensor
    starts_at:torch.Tensor
    ends_at:torch.Tensor
    minlen:int
    maxlen:int
    idx: int = 0
    dispatch:List[callable]
    sessions:List[uuid.UUID]

    def __init__(self, states: List[HelloSippyPipeState], pp:HelloSippyRTPipe):
        self.merge(states, pp)

    def merge(self, states:List[HelloSippyPipeState], pp:HelloSippyRTPipe):
        self.dispatch = [s.dispatch for s in states]
        max_statelen = max([x.encoder_attention_mask.size(1) for x in states])
        for aname in ('inputs', 'speaker_embeddings', 'encoder_attention_mask', 'pre_frames', 'starts_at', 'ends_at'):
            aval = [getattr(s, aname) for s in states]
            if aname in ('inputs', 'encoder_attention_mask'):
                for ia in [ia for ia, a in enumerate(aval) if a.size(1) < max_statelen]:
                    new_size = list(aval[ia].size())
                    new_size[1] = max_statelen
                    aval[ia] = pad_tensor_to_target(aval[ia], new_size)
            #print(f'{aname=} {[x.shape for x in aval]=}')
            setattr(self, aname, torch.cat(aval).contiguous())
        encoder_out = pp.model.speecht5.encoder(
            input_values=self.inputs,
            attention_mask=self.encoder_attention_mask,
            return_dict=True,
        )
        self.encoder_last_hidden_state = encoder_out.last_hidden_state
        self.maxlen = int(self.encoder_last_hidden_state.size(1) * pp.maxlenratio / pp.model.config.reduction_factor)
        self.minlen = int(self.encoder_last_hidden_state.size(1) * pp.minlenratio / pp.model.config.reduction_factor)

        # Start the output sequence with a mel spectrum that is all zeros.
        self.output_sequence = self.encoder_last_hidden_state.new_zeros(self.inputs.size(0), 1, pp.model.config.num_mel_bins)
        #if self.past_key_values is not None:
        #        batch_size = self.speaker_embeddings.size(0)
        #        past_key_values = [list(x) for x in self.past_key_values]
        #        for past_key_value, idx, t in [(x, i, _x) for x in past_key_values for i, _x in enumerate(x)]:
        #            new_size = list(t.size())
        #            new_size[0] = batch_size
        #            if idx >= 2: new_size[2] = max_statelen
        #            #new_size[-1] = max_statelen
        #            assert id(past_key_value[idx]) == id(t)
        #            past_key_value[idx] = pad_tensor_to_target(t, new_size)
        #            #raise Exception(f"FIXME: NOT IMPLEMENTED: {past_key_value[idx].shape=}")
        #        self.past_key_values = tuple([tuple(x) for x in past_key_values])
        #if self.past_key_values is not None: print(f"FIXME: NOT IMPLEMENTED: {self.past_key_values[0][1].shape=} {self.past_key_values[1][0].shape=}")
        #self.past_key_values = None
        #print(f'{self.dispatch=}')

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
        if period > 0.0: self.elp = ElPeriodic(1.0 / period)
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

    def __del__(self):
        print('del')
        if not hasattr(self, 'queue'): return
        self.queue.put(None)
        self.join()
        self.func = None

import torchaudio.transforms as T

class WeakDispatcher():
    def __init__(self, queue:Queue): self.queue = weakref.ref(queue)
    def __call__(self, res):
        q = self.queue()
        if q: q.put(res.numpy() if res is not None else None)

class InfernSession:
    _cmd_queue:Queue
    id:uuid.UUID
    default_speaker:torch.Tensor
    def __init__(self, queue, default_speaker:torch.Tensor): self.id, self._cmd_queue, self.default_speaker = uuid.uuid4(), queue, default_speaker
    def play(self, text:str, dispatch:Queue, speaker:Optional[torch.Tensor] = None):
        cmd = HelloSippyPlayRequest(self.id, text, speaker if speaker else self.default_speaker, WeakDispatcher(dispatch))
        self._cmd_queue.put(cmd)

class HelloSippyRTPipe:
    processor: SpeechT5Processor
    model: SpeechT5ForTextToSpeech
    chunker: AmendmentNetwork1
    resampler: Optional[T.Resample]
    minlenratio: float = 0.0
    maxlenratio: float = 20.0
    threshold: float = 0.5
    chunk_size: int = 8
    pre_nframes: int = 2
    post_nframes: int = 2
    model_sr: int = 16000
    output_sr: int = 16000
    default_model = "microsoft/speecht5_tts"

    def __init__(self, device, model=default_model, get_processor:Optional[callable]=None, output_sr:int=output_sr, **kwa):
        self.cuda_lock = InfernGlobals().torcher
        with self.cuda_lock:
            if get_processor is None:
               self.processor = SpeechT5Processor.from_pretrained(model)
            else:
                self.processor = get_processor(device, model)
            mc = SpeechT5Config.from_pretrained(model, max_speech_positions=4000, **kwa)
            model = maybe_half(SpeechT5ForTextToSpeech.from_pretrained(model,
                                                            config=mc)).to(device)
            model.speecht5.decoder = maybe_half(model.speecht5.decoder)
            model.speecht5.encoder = maybe_half(model.speecht5.encoder)
            model.eval()
            self.model = model
            _vc_conf = SpeechT5HifiGanConfig()
            vocoder = maybe_half(SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",
                                                    config = _vc_conf)).to(device)
            vocoder.eval()
            self.vocoder = vocoder
            self.c_conf = AmendmentNetwork1Config()
            chunker = AmendmentNetwork1.from_pretrained("sobomax/speecht5-rt.post_vocoder.v2",
                                                        config=self.c_conf)
            chunker = maybe_half(chunker).to(device)
            chunker.eval()
            self.chunker = chunker
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            self.speaker_embeddings = [torch.tensor(ed["xvector"], device='cpu').unsqueeze(0)
                                        for ed in sorted(embeddings_dataset, key=lambda x: x['filename'])]
            for x in [_x for x in (self.model.parameters, self.vocoder.parameters, self.chunker.parameters) for _x in x()] + self.speaker_embeddings: x.requires_grad = False
            if self.model_sr != output_sr:
                self.resampler = maybe_half(T.Resample(orig_freq=self.model_sr, new_freq=output_sr)).to(device)
            else:
                self.resampler = None
            self.output_sr = output_sr

    def infer(self, state:HelloSippyPipeStateBatched) -> None:
        with self.cuda_lock:
            batch_size = state.output_sequence.size(0)
            spectrogram = maybe_half(torch.zeros(batch_size, 0, self.model.config.num_mel_bins)).to(self.model.device)
            while spectrogram.size(1) < (self.chunk_size * 4):
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
                                             state.idx+(self.pre_nframes+self.post_nframes)//self.model.config.reduction_factor, state.ends_at)
                state.idx += 1
            spectrogram = self.model.speech_decoder_postnet.postnet(spectrogram)
            spectrogram = torch.cat((state.pre_frames, spectrogram), dim=1)
            eframes = self.pre_nframes + self.post_nframes
            state.pre_frames = spectrogram[:, -eframes:, :]
            nchunks = spectrogram.size(1) // self.chunk_size
            spectrogram = torch.cat([spectrogram[:, i*self.chunk_size:(i+1)*self.chunk_size+eframes, :] for i in range(nchunks)], dim=0)
            audio = self.vocoder(spectrogram)
            audio = self.chunker(spectrogram, audio)
            slices = audio.split(batch_size, dim=0)
            audio = torch.cat(slices, dim=1)
            state.audio = self.resampler(audio) if self.resampler else audio

    def unbatch_and_dispatch(self, state:HelloSippyPipeStateBatched):
        audio, sr_rr = state.audio, self.model_sr // self.output_sr
        end_idx = state.idx - 1
        stepsize = 256 * 2 // sr_rr
        with self.cuda_lock:
            for i, dispatch in [(i, _cbq) for i, _cbq in enumerate(state.dispatch) if _cbq is not None]:
                startoff = max(0, (asize:=audio[i].size(0)) - ((state.idx - state.starts_at[i].item()) * stepsize))
                endoff = min(asize, asize - (((state.idx - ends_at) * stepsize) if (ends_at:=state.ends_at[i].item()) >=0 else 0))
                assert startoff <= endoff
                if startoff != endoff:
                    dispatch(audio[i][startoff:endoff].to(torch.float16).cpu())
                if ends_at >= 0 and ends_at <= end_idx:
                    dispatch(None)
                    state.dispatch[i] = None
            mask = ((state.ends_at < 0) | (state.ends_at > end_idx))
            if torch.all(~mask):
                return False
        return True

    def get_rand_voice(self):
        s_index = torch.randint(0, len(self.speaker_embeddings), (1,)).item()
        rv = self.speaker_embeddings[s_index]
        return rv

    #@lru_cache(maxsize=16)
    def get_voice(self, s_index:int):
        rv = self.speaker_embeddings[s_index]
        return rv

class HelloSippyRTPipeTest(HelloSippyRTPipe):
    _main_thread_id: int
    _sync_queue: Queue
    sessions: weakref.WeakValueDictionary[InfernSession]
    max_sessions: int = 50
    output_sr = 8000

    def __init__(self, *a, **kwa):
        self._main_thread_id = threading.get_ident()
        self._sync_queue = Queue()
        self.sessions = weakref.WeakValueDictionary()
        super().__init__(*a, **kwa)

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

    class WorkerState: state:Optional[HelloSippyPipeStateBatched]=None; live:Optional[List[uuid.UUID]]=None

    @trp_thread(noreturn=True)
    def synchronize(self, ws:Optional[WorkerState]) -> None:
        if not ws: ws = self.WorkerState()
        state = ws.state
        if state: return (self.main_gen(ws), None)[-1]
        ssq = [self._sync_queue.get(),]
        try:
            while True: ssq.append(self._sync_queue.get_nowait())
        except QueueEmpty: pass
        assert all(isinstance(x, SessCmd) for x in ssq)
        syncs, reqs = [x for x in ssq if isinstance(x, SessSyncCmd)], [x for x in ssq if not isinstance(x, SessSyncCmd)]
        if len(syncs) == 0 and len(reqs) == 0: raise AssertionError(f'this could not be happening {ssq=}')
        #print(f'{len(syncs)=} {len(reqs)=} {syncs=}')
        ws.live = live = syncs[-1].live if len(syncs) > 0 else ws.live
        if not live: return (self.synchronize(ws), None)[-1]
        reqs_live = [x for x in reqs if x.session in live]
        if len(reqs_live) == 0: return (self.synchronize(ws), None)[-1]
        with self.cuda_lock:
            new_states = [HelloSippyPipeState(self, r) for r in reqs_live]
            #if state: state.mergein(new_states)
            ws.state = HelloSippyPipeStateBatched(new_states, self)
        #raise Exception(f'{len(ssq)=} {reqs_live=} {live=} {len(self.sessions)=}')
        self.main_gen(ws)

    @trp_thread(noreturn=True)
    def main_gen(self, ws:WorkerState) -> None:
        super().infer(ws.state)
        #print(f'{state.ends_at.shape=} {state.ends_at.cpu().numpy()=} {state.audio.shape=}')
        self.unbatch_and_dispatch(ws)

    @trp_thread(noreturn=True)
    def unbatch_and_dispatch(self, ws:WorkerState):
        more = super().unbatch_and_dispatch(ws.state)
        if not more:
            ws.state = None
        self.synchronize(ws)

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
    from time import monotonic
    seed_RNGs()
    from random import choices
    from utils.tts import smith_set, bender_set, hal_set
    n = 50
    prompts = choices([y for x in smith_set() + bender_set() + hal_set() for y in x.split('|')], k=n)
    #prompts = prompts
    #prompts = [prompts[0] for _ in range(n)]
    @dataclass(frozen=True)
    class ResFeedback: n:int; time_to_first_frame:float; time_to_last_frame:float; number_of_frames:int
    class res_cb(threading.Thread):
        def __init__(self, n, name='dispatch', res_queue=None):
            super().__init__(target=self.__thread)
            self.n, self.name, self.res_queue = n, name, res_queue
            if self.name == 'dispatch': self.data = np.empty(0)
            self.q = Queue()
            self.daemon = True
            self.start()

        def __thread(self):
            st = monotonic()
            time_to_first_frame = None
            while (y:=self.q.get()) is not None:
                #print(f'{self.name}{self.n}({y.shape=})')
                self.data = np.concatenate((self.data, y), axis=0)
                if time_to_first_frame is None: time_to_first_frame = monotonic() - st
            self.eos(ResFeedback(self.n, time_to_first_frame, monotonic()-st, int(self.data.shape[0])))

        def eos(self, res:ResFeedback):
            sys.stdout.write(f'eos({self.n}) {self.data.shape=}\n')
            sys.stdout.flush()
            sf.write(f'out_{self.n}.wav', self.data, 8000, 'PCM_16')
            if self.res_queue: self.res_queue.put(res)

    params = {'hidden_dropout':0.0, 'positional_dropout':0.0, 'speech_decoder_prenet_dropout':0.0,
              'activation_dropout':0.0, 'encoder_layerdrop':0.0, 'decoder_layerdrop':0.0, 'attention_dropout':0.0,
              'speech_decoder_postnet_dropout':0.0, 'feat_proj_dropout':0.0}
    sp = HelloSippyRTPipeTest('xpu')
    if ipex is not None:
        sp.model = ipex.optimize(sp.model, dtype=torch.bfloat16)
        sp.vocoder = ipex.optimize(sp.vocoder, dtype=torch.bfloat16)
        sp.chunker = ipex.optimize(sp.chunker, dtype=torch.bfloat16)

    s1 = [sp.alloc_session() for i in range(50)]
    del s1
    res_queue = Queue()
    from time import sleep
    #sp.synchronize(None)
    s2 = [((s:=sp.alloc_session()), (r:=res_cb(n, res_queue=res_queue)), s.play(p, r.q), 'sleep(0.5)') for n, p in enumerate(prompts)]
    sp.synchronize(None)
    for _ in range(len(s2)):
        res = res_queue.get()
        rtr = (res.time_to_last_frame - res.time_to_first_frame) / (res.number_of_frames / 8000)
        print(f'Sess#{res.n}: {res.time_to_first_frame=}, {res.time_to_last_frame=}, {res.number_of_frames=} {rtr=}')
        sys.stdout.flush()
        s2[res.n][1].join()
    return(0)

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
