try: import intel_extension_for_pytorch as ipex
except ModuleNotFoundError: ipex = None

from typing import Dict, List, Optional, Any, Tuple
from queue import Queue
from uuid import UUID, uuid4
from functools import partial
from fractions import Fraction
import pickle
import gzip
from time import monotonic

import ray
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from tensorboardX import SummaryWriter

from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernSTTActor import InfernSTTActor
from Cluster.STTSession import STTRequest, STTResult
from Core.T2T.Translator import Translator
from RTP.RTPOutputWorker import TTSSMarkerNewSent

from utils.tts import smith_set, bender_set, hal_set


def get_embedding(t, m, sentence):
    with torch.no_grad():
        inputs = t(sentence, return_tensors='pt', padding=True, truncation=True).to('xpu')
        outputs = m(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Squeeze to remove batch dimension

class SoundPreBatcher():
    audio: Optional[torch.Tensor] = None
    deliver: callable
    stt_done: callable
    tts_done: callable
    lang: str
    def __init__(self, deliver, stt_done:callable, tts_done:callable, lang:str):
        self.deliver = deliver
        self.stt_done = stt_done
        self.tts_done = tts_done
        self.lang = lang

    def __call__(self, chunk):
        if not isinstance(chunk, TTSSMarkerNewSent):
            audio = self.audio if self.audio is not None else []
            audio.append(chunk.audio)
            self.audio = audio
            #print(f'audio: {ts.audio.size()=}')
        else:
            nid = ray.get_runtime_context().get_node_id()
            self.tts_done(nid)
            self.deliver(STTRequest(torch.cat(self.audio).numpy(), self.stt_done, self.lang))
            self.audio = None
        return (0, False)

class TestPipe():
    id: UUID
    def __init__(self, tts_actr, stt_actr, bench_actr, lang):
        self.id = uuid4()
        self.tts_sess = ray.get(tts_actr.new_tts_session.remote())
        stt_done = partial(bench_actr.stt_done.remote, session_id=self.id)
        tts_done = bench_actr.tts_done.remote
        self.stt_sess = ray.get(stt_actr.new_stt_session.remote())
        self.stt_soundin = partial(stt_actr.stt_session_soundin.remote, self.stt_sess)
        tts_soundout = SoundPreBatcher(self.stt_soundin, stt_done, tts_done, lang)
        ray.get(tts_actr.tts_session_start.remote(self.tts_sess, tts_soundout))
        self.tts_say = partial(tts_actr.tts_session_say.remote, rgen_id=self.tts_sess)
        self.lang = lang

class BEval():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased').to('xpu')

    def __call__(self, prompt, text, embedding1=None):
        if embedding1 is None:
            embedding1 = get_embedding(self.tokenizer, self.model, prompt)
        embedding2 = get_embedding(self.tokenizer, self.model, text)
        return 1.0 - cosine(embedding1, embedding2)

class TestSession():
    id: UUID
    ver: int = 0
    prompts: List[str]
    tts_sess_id: UUID
    stt_sess_id: UUID
    speaker_id: int
    nres: int = 0
    tot_error: float = 0.0
    tot_duration: Fraction
    results: List[Tuple[str, STTResult]]
    def __init__(self, prompts, speaker_id, test_pipe, beval):
        self.id = uuid4()
        self.prompts = list(prompts)
        self.speaker_id = speaker_id
        self.test_pipe = test_pipe
        self.beval = beval
        self.audio = torch.zeros(0)
        self.tot_duration = Fraction(0)
        self.results = []

    def next_prompt(self, last_res:Optional[STTResult]=None):
        play_id = 0 if last_res is None else 1
        if len(self.prompts) > play_id:
            res = self.test_pipe.tts_say(text=self.prompts[play_id][0], speaker_id=self.speaker_id)
            ray.get(res)
        if play_id > 0:
            prompt = self.prompts.pop(0)
            similarity = self.beval(prompt[0], last_res.text, prompt[1])
            self.nres += 1
            assert similarity >= 0.0 and similarity <= 1.0
            assert last_res.no_speech_prob >= 0.0 and last_res.no_speech_prob <= 1.0
            self.tot_error = max(self.tot_error, max((1.0 - similarity), last_res.no_speech_prob))
            self.tot_duration += last_res.duration
            #self.tot_error += 1.0 - similarity
            print(f"Speaker[{self.speaker_id}]: max_error={self.max_error()}")
            self.results.append((prompt, last_res))
            self.save()
        return True if self.prompts else False

    def save(self):
        _t = TestSession(self.prompts, self.speaker_id, None, None)
        _t.tot_error = self.tot_error
        _t.tot_duration = self.tot_duration
        _t.nres = self.nres
        _t.results = self.results
        lang = self.test_pipe.lang
        with gzip.open(f'checkpoint/{lang}/speaker.{self.speaker_id}.{lang}.pkl.gz', 'wb') as file:
            pickle.dump(_t, file)

    def max_error(self):
        return self.tot_error

def load_checkpoints(lang, gen=None):
    i = 0
    res = []
    skips = 0
    while True:
        try:
            with gzip.open(f'checkpoint/{lang}/speaker.{i}.{lang}.pkl.gz', 'rb') as file:
                s = pickle.load(file)
                assert len(s.results) == s.nres
                res.append(s)
        except FileNotFoundError:
            skips += 1
            if skips > 2000: break
        i += 1
    if gen is None:
        gen = max(r.nres for r in res)
    res = [r for r in res if r.nres >= gen]
    return res, gen

import sys

def reeval(res, beval, gen, prompts):
    for g in range(gen):
        _prompt = prompts[g]
        prompt, embedding = _prompt[0], _prompt[1]()
        for i, r in enumerate(res):
            lr = r.results[g]
            if not isinstance(lr, STTResult):
                prompt_orig, lr = r.results[g]
                if isinstance(prompt_orig, tuple):
                    prompt_orig = prompt_orig[0]
                    r.results[g] = (prompt_orig, lr)
                assert prompt_orig == prompt, f'{prompt_orig=} != {prompt=}'
            else:
                r.results[g] = (prompt, lr)
            if lr.text[0] == ' ': lr.text = lr.text[1:]
            similarity = beval(prompt, lr.text, embedding)
            tot_error = max((1.0 - similarity), lr.no_speech_prob)
            if r.nres == g: assert abs(tot_error - r.tot_error) < 0.000001, f"Speaker[{r.speaker_id}]: {tot_error=} {r.tot_error=}"
            r.tot_error = max(tot_error, (0 if g == 0 else r.tot_error))
            r.tot_duration = lr.duration + (0 if g == 0 else r.tot_duration)
            sys.stdout.write(f'Reeval@{g}: {i} of {len(res)}\r')
        sys.stdout.write('\n')
    for r in res:
        r.nres = gen
        r.results = r.results[:gen]
        r.beval = beval

def plotdensity(res, fname='density_plot'):
    import matplotlib.pyplot as plt
    plt.figure()
    errors = [x.tot_error for x in res]
    plt.hist(errors, bins=400)
    plt.savefig(f'{fname}.png', format='png', dpi=300)

class TestPipes():
    ntts:Optional[Dict[ray.actor, int]] = None
    frst_tts:Dict[ray.actor, float]
    last_tts:Dict[ray.actor, float]
    def __init__(self, ntts_a, nstt_a, npipes, tts_lang, bench_actor):
        self.ntts_a, self.nstt_a, self.npipes,self.tts_lang, self.bench_actor = ntts_a, nstt_a, npipes, tts_lang, bench_actor
        self.setup()

    def setup(self):
        self.tts_actrs = tuple(InfernTTSActor.remote() for _ in range(self.ntts_a))
        self.stt_actrs = tuple(InfernSTTActor.remote() for _ in range(self.nstt_a))
        fut = tuple(x.start.remote() for x in self.stt_actrs)
        fut += tuple(x.start.remote(self.tts_lang) for x in self.tts_actrs)
        def stt_actr(_s): return  self.stt_actrs[_s % len(self.stt_actrs)]
        def tts_actr(_s): return  self.tts_actrs[_s % len(self.tts_actrs)]
        self.test_pipes = list(TestPipe(tts_actr(i), stt_actr(i), self.bench_actor, self.tts_lang)
                          for i in range(self.npipes))
        for x in fut: ray.get(x)

    def reset(self):
        otts_actrs = self.tts_actrs
        for actr in self.tts_actrs + self.stt_actrs:
            ray.kill(actr)
        self.setup()
        #for oa, na in zip(otts_actrs, self.tts_actrs):
        #    self.frst_tts[na] = self.frst_tts[oa]
        #    self.last_tts[na] = self.last_tts[oa]
        #    del self.frst_tts[oa], self.last_tts[oa]

@ray.remote(resources={"head": 1})
class InfernBenchActor():
    default_resources = {'head':1, 'stt': 3, 'tts':3}
    queue: Queue
    sessions: Dict[int, TestSession]
    def __init__(self, _): pass

    def loop(self):
        self.writer = SummaryWriter()
        lang = 'it'
        self_actor = ray.get_runtime_context().current_actor
        batch_size = (self.default_resources['stt'] + self.default_resources['tts']) * 8
        self.TPS = TPS = TestPipes(self.default_resources['tts'], self.default_resources['stt'], batch_size, lang, self_actor)
        _prompts = [y for x in smith_set() + bender_set() + hal_set() for y in x.split('|')]
        if lang != 'en':
            tr = Translator('en', lang)
            _prompts = [tr.translate(p.strip()) for x in _prompts for p in x.split('|')]
        _prompts.sort(key=lambda x: len(x), reverse=True)
        _prompts.pop(0)
        beval = BEval()
        prompts = [(x, partial(get_embedding, beval.tokenizer, beval.model, x)) for x in _prompts]
        gen = 7
        res, _gen = load_checkpoints(lang, gen+1)
        reeval(res, beval, gen+1, prompts)
        plotdensity(res)
        cut_trs = 0.10
        #raise Exception(f'{len(res)=}')
        self.queue = Queue()
        self.sessions = {}
        target_nspeakers = 300
        class _next_ts_0():
            speaker_id = 0
            def __init__(self, beval):
                self.beval = beval
            def __call__(self, tp, prompt):
                prompt = list(prompt)
                try:
                    with gzip.open(f'checkpoint/speaker.{self.speaker_id}.{tp.lang}.pkl.gz', 'rb') as file:
                        ts = pickle.load(file)
                    ts.prompts = prompt
                    ts.test_pipe = tp
                    ts.beval = self.beval
                except FileNotFoundError:
                    ts = TestSession(prompt, self.speaker_id, tp, self.tokenizer, self.model)
                self.speaker_id += 1
                return ts
        class _next_ts():
            def __init__(self, res):
                self.res = res
            def __call__(self, tp, prompt):
                ts = self.res.pop(0)
                ts.prompts = list(prompt)
                ts.test_pipe = tp
                return ts
        next_ts, draining = (_next_ts_0(beval), False) if _gen == 0 else (_next_ts(res), True)
        prompt = None
        while True:
            if prompt is None: prompt = tuple((x, y()) for x, y in prompts[gen:gen+1])
            if len(TPS.test_pipes) > 0 and not draining:
                tp = TPS.test_pipes.pop()
                try:
                    while (ts:=next_ts(tp, prompt)) and ts.nres > gen:
                        print(f"pre-loaded Speaker[{ts.speaker_id}]: max_error={ts.max_error()}")
                        res.append(ts)
                    if ts.nres > 0 and ts.nres < gen:
                        self.queue.put(ts)
                        if ts.nres > 0: print(f"Previously disqualified Speaker[{ts.speaker_id}]@{ts.nres}: max_error={ts.max_error()}")
                        continue
                    assert ts.next_prompt()
                except IndexError:
                    TPS.test_pipes.append(tp)
                    draining = True
                    continue
                self.sessions[tp.id] = ts
                #if ts.speaker_id > 2000:
                #    draining = True
            while len(TPS.test_pipes) == 0 or (draining and len(TPS.test_pipes) < batch_size):
                if draining:
                    print(f'Draining: {batch_size - len(TPS.test_pipes)} left')
                ts = self.queue.get()
                res.append(ts)
                TPS.test_pipes.append(ts.test_pipe)
            if draining:
                #for t in tts_actrs: ray.get(t.stop.remote())
                #raise Exception("BP")
                plotdensity(res, f'density_plot_end_{gen}')
                ilen = len(res)
                res = [r for r in res if r.tot_error < cut_trs]
                plotdensity(res, f'density_plot_start_{gen+1}')
                print(f'Cutting from {ilen} to {len(res)}')
                if len(res) <= target_nspeakers:
                    break
                next_ts = _next_ts(res)
                res = []
                prompt = None
                draining = False
                gen += 1
                TPS.reset()

        for ts in res:
            print(f"Speaker[{ts.speaker_id}]: max_error={ts.max_error()}")

    def stt_done(self, session_id, result:STTResult):
        ts = self.sessions[session_id]
        print(f'text: {result.text}, no_speech_prob: {result.no_speech_prob}')
        more = ts.next_prompt(result)
        if not more:
            del self.sessions[session_id]
            self.queue.put(ts)
        #if not self.sessions: self.stop()

    def tts_done(self, nid):
        tstp = monotonic()
        if self.TPS.ntts is None:
            self.TPS.ntts, self.TPS.last_tts, self.TPS.frst_tts = {}, {}, {}
        if nid not in self.TPS.ntts:
            self.TPS.frst_tts[nid] = self.TPS.last_tts[nid] = tstp
            self.TPS.ntts[nid] = 1
        else:
            nact = tuple(self.TPS.ntts.keys()).index(nid)
            ival = tstp - self.TPS.frst_tts[nid]
            ntts = self.TPS.ntts[nid]
            sntts = sum(self.TPS.ntts.values())
            self.writer.add_scalar(f'tts/rate_{nact}', ntts / ival, sntts)
            self.writer.flush()
            self.TPS.ntts[nid] += 1
        self.TPS.last_tts[nid] = tstp

    def stop(self):
        self.queue.put(None)
        self.writer.close()
