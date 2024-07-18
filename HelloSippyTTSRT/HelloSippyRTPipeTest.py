try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

import sys, random, weakref, uuid
from typing import List, Optional, Tuple
import contextlib, time
from os.path import exists as path_exists
from queue import Queue, Empty as QueueEmpty
from dataclasses import dataclass

import numpy as np

import torch
for i in range(2):
    try:
        from HelloSippyTTSRT.HelloSippyRTPipe import HelloSippyRTPipe, HelloSippyPipeState, HelloSippyPipeStateBatched, \
        HelloSippyPlayRequest, SessCmd, SessSyncCmd
    except ModuleNotFoundError:
        from sys import path as sys_path
        from os import getcwd
        sys_path.append(getcwd())
    else: break
else: raise ModuleNotFoundError('HelloSippyRTPipe')

from transformers import set_seed

class ErrMaxSessReached(Exception): pass

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

class WeakDispatcher():
    def __init__(self, queue:Queue): self.queue = weakref.ref(queue)
    def __call__(self, res):
        q = self.queue()
        if q: q.put(res.to(torch.float16).numpy() if res is not None else None)

class InfernSession:
    _cmd_queue:Queue
    id:uuid.UUID
    default_speaker:torch.Tensor
    def __init__(self, queue, default_speaker:torch.Tensor): self.id, self._cmd_queue, self.default_speaker = uuid.uuid4(), queue, default_speaker
    def play(self, text:str, dispatch:Queue, speaker:Optional[torch.Tensor] = None):
        cmd = HelloSippyPlayRequest(self.id, text, speaker if speaker else self.default_speaker, WeakDispatcher(dispatch))
        self._cmd_queue.put(cmd)

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
        if not speaker: speaker = self.get_rand_voice()[0]
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
    sp = HelloSippyRTPipeTest('xpu' if ipex is not None else 'cuda')
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
