from typing import Dict, List, Optional
from queue import Queue
from uuid import UUID, uuid4
from functools import partial
from fractions import Fraction
import pickle
import gzip

import ray
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernSTTActor import InfernSTTActor
from Cluster.STTSession import STTRequest, STTResult
from Core.T2T.Translator import Translator
from TTSRTPOutput import TTSSMarkerNewSent, TTSSMarkerEnd

from utils.tts import smith_set, bender_set, hal_set

def get_embedding(t, m, sentence):
    inputs = t(sentence, return_tensors='pt', padding=True, truncation=True)
    outputs = m(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()  # Squeeze to remove batch dimension

class SoundPreBatcher():
    audio: Optional[torch.Tensor] = None
    deliver: callable
    stt_done: callable
    lang: str
    def __init__(self, deliver, stt_done:callable, lang:str):
        self.deliver = deliver
        self.stt_done = stt_done
        self.lang = lang

    def __call__(self, chunk):
        if not isinstance(chunk, TTSSMarkerNewSent):
            audio = self.audio if self.audio is not None else []
            audio.append(chunk)
            self.audio = audio
            #print(f'audio: {ts.audio.size()=}')
        else:
            self.deliver(STTRequest(torch.cat(self.audio).numpy(), self.stt_done, self.lang))
            self.audio = None
        return (0, False)

class TestPipe():
    id: UUID
    def __init__(self, tts_actr, stt_actr, stt_done, lang):
        self.id = uuid4()
        self.tts_sess = ray.get(tts_actr.new_tts_session.remote())
        stt_done = partial(stt_done, session_id=self.id)
        self.stt_sess = ray.get(stt_actr.new_stt_session.remote())
        self.stt_soundin = partial(stt_actr.stt_session_soundin.remote, self.stt_sess)
        tts_soundout = SoundPreBatcher(self.stt_soundin, stt_done, lang)
        ray.get(tts_actr.tts_session_start.remote(self.tts_sess, tts_soundout))
        self.tts_say = partial(tts_actr.tts_session_say.remote, rgen_id=self.tts_sess)
        self.lang = lang

class TestSession():
    id: UUID
    prompts: List[str]
    tts_sess_id: UUID
    stt_sess_id: UUID
    speaker_id: int
    nres: int = 0
    tot_error: float = 0.0
    tot_duration: Fraction
    results: List[STTResult]
    def __init__(self, prompts, speaker_id, test_pipe, tokenizer, model):
        self.id = uuid4()
        self.prompts = list(prompts)
        self.speaker_id = speaker_id
        self.test_pipe = test_pipe
        self.tokenizer = tokenizer
        self.model = model
        self.audio = torch.zeros(0)
        self.tot_duration = Fraction(0)
        self.results = []

    def next_prompt(self, last_res:Optional[STTResult]=None):
        play_id = 0 if last_res is None else 1
        if len(self.prompts) > play_id:
            res = self.test_pipe.tts_say(text=self.prompts[play_id][0], speaker_id=self.speaker_id)
            ray.get(res)
        if play_id > 0:
            embedding1 = self.prompts.pop(0)[1]
            embedding2 = get_embedding(self.tokenizer, self.model, last_res.text)
            similarity = 1 - cosine(embedding1, embedding2)
            self.nres += 1
            assert similarity >= 0.0 and similarity <= 1.0
            assert last_res.no_speech_prob >= 0.0 and last_res.no_speech_prob <= 1.0
            self.tot_error += max((1.0 - similarity), last_res.no_speech_prob)
            self.tot_duration += last_res.duration
            #self.tot_error += 1.0 - similarity
            print(f"Speaker[{self.speaker_id}]: average_error={self.average_error()}")
            self.results.append(last_res)
            self.save()
        return True if self.prompts else False

    def save(self):
        _t = TestSession(self.prompts, self.speaker_id, None, None, None)
        _t.tot_error = self.tot_error
        _t.tot_duration = self.tot_duration
        _t.nres = self.nres
        _t.results = self.results
        with gzip.open(f'checkpoint/speaker.{self.speaker_id}.{self.test_pipe.lang}.pkl.gz', 'wb') as file:
            pickle.dump(_t, file)

    def average_error(self):
        return self.tot_error / self.nres

@ray.remote(resources={"head": 1})
class InfernBenchActor():
    default_resources = {'head':1, 'stt': 3, 'tts':3}
    queue: Queue
    sessions: Dict[int, TestSession]
    def __init__(self, _): pass

    def loop(self):
        lang = 'de'
        tts_actrs = tuple(InfernTTSActor.remote() for _ in range(self.default_resources['tts']))
        stt_actrs = tuple(InfernSTTActor.remote() for _ in range(self.default_resources['stt']))
        fut = tuple(x.start.remote() for x in stt_actrs)
        fut += tuple(x.start.remote(lang) for x in tts_actrs)
        _prompts = [y for x in smith_set() + bender_set() + hal_set() for y in x.split('|')]
        if lang != 'en':
            tr = Translator('en', lang)
            _prompts = [tr.translate(x) for x in _prompts]
        _prompts.sort(key=lambda x: len(x), reverse=True)
        _prompts.pop(0)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        self.queue = Queue()
        self.sessions = {}
        for x in fut: ray.get(x)
        self_actor = ray.get_runtime_context().current_actor
        prompts = [(x, partial(get_embedding, tokenizer, model, x)) for x in _prompts]
        def stt_actr(_s): return  stt_actrs[_s % len(stt_actrs)]
        def tts_actr(_s): return  tts_actrs[_s % len(tts_actrs)]
        res = []
        batch_size = 48 + 4
        target_nspeakers = 30
        test_pipes = list(TestPipe(tts_actr(i), stt_actr(i), self_actor.stt_done.remote, lang)
                          for i in range(batch_size))
        draining = False
        class _next_ts():
            speaker_id = 0
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model
            def __call__(self, tp, prompt):
                prompt = list(prompt)
                try:
                    with gzip.open(f'checkpoint/speaker.{self.speaker_id}.{tp.lang}.pkl.gz', 'rb') as file:
                        ts = pickle.load(file)
                    ts.prompts = prompt
                    ts.test_pipe = tp
                    ts.tokenizer = self.tokenizer
                    ts.model = self.model
                except FileNotFoundError:
                    ts = TestSession(prompt, self.speaker_id, tp, self.tokenizer, self.model)
                self.speaker_id += 1
                return ts
        next_ts = _next_ts(tokenizer, model)
        prompt = None
        gen = 0
        while True:
            if prompt is None: prompt = tuple((x, y()) for x, y in prompts[gen:gen+1])
            if len(test_pipes) > 0 and not draining:
                tp = test_pipes.pop()
                try:
                    while (ts:=next_ts(tp, prompt)) and ts.nres > gen:
                        print(f"pre-loaded Speaker[{ts.speaker_id}]: average_error={ts.average_error()}")
                        res.append(ts)
                    if ts.nres > 0 and ts.nres < gen:
                        self.queue.put(ts)
                        if ts.nres > 0: print(f"Previously disqualified Speaker[{ts.speaker_id}]@{ts.nres}: average_error={ts.average_error()}")
                        continue
                    assert ts.next_prompt()
                except IndexError:
                    test_pipes.append(tp)
                    draining = True
                    continue
                self.sessions[tp.id] = ts
            while len(test_pipes) == 0 or (draining and len(test_pipes) < batch_size):
                if draining: print(f'Draining: {batch_size - len(test_pipes)} left')
                ts = self.queue.get()
                res.append(ts)
                test_pipes.append(ts.test_pipe)
            if draining:
                res.sort(key=lambda x: x.average_error())
                if (len(res) / 2) <= target_nspeakers:
                    res = res[:target_nspeakers]
                    break
                res = res[:int(len(res)/2)]
                class _next_ts():
                    def __init__(self, res):
                        self.res = res
                    def __call__(self, tp, prompt):
                        ts = self.res.pop(0)
                        ts.prompts = list(prompt)
                        ts.test_pipe = tp
                        return ts
                next_ts = _next_ts(res)
                res = []
                prompt = None
                draining = False
                gen += 1

        for ts in res:
            print(f"Speaker[{ts.speaker_id}]: average_error={ts.average_error()}")

    def stt_done(self, session_id, result:STTResult):
        ts = self.sessions[session_id]
        print(f'text: {result.text}, no_speech_prob: {result.no_speech_prob}')
        more = ts.next_prompt(result)
        if not more:
            del self.sessions[session_id]
            self.queue.put(ts)
        #if not self.sessions: self.stop()

    def stop(self):
        self.queue.put(None)
