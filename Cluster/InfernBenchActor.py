from typing import Dict, List, Optional
from queue import Queue
from uuid import UUID, uuid4
from functools import partial

import ray
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine

from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernSTTActor import InfernSTTActor
from TTSRTPOutput import TTSSMarkerNewSent, TTSSMarkerEnd

from utils.tts import smith_set, bender_set, hal_set

def get_embedding(t, m, sentence):
    inputs = t(sentence, return_tensors='pt', padding=True, truncation=True)
    outputs = m(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()  # Squeeze to remove batch dimension

class SoundPreBatcher():
    audio: Optional[torch.Tensor] = None
    deliver: callable
    def __init__(self, deliver):
        self.deliver = deliver

    def __call__(self, chunk):
        if not isinstance(chunk, TTSSMarkerNewSent):
            audio = self.audio if self.audio is not None else []
            audio.append(chunk)
            self.audio = audio
            #print(f'audio: {ts.audio.size()=}')
        else:
            self.deliver(torch.cat(self.audio).numpy())
            self.audio = None
        return (0, False)

class TestPipe():
    id: UUID
    def __init__(self, tts_actr, stt_actr, stt_done):
        self.id = uuid4()
        self.tts_sess = ray.get(tts_actr.new_tts_session.remote())
        stt_done = partial(stt_done, session_id=self.id)
        self.stt_sess = ray.get(stt_actr.new_stt_session.remote(stt_done))
        self.stt_soundin = partial(stt_actr.stt_session_soundin.remote, self.stt_sess)
        tts_soundout = SoundPreBatcher(self.stt_soundin)
        ray.get(tts_actr.tts_session_start.remote(self.tts_sess, tts_soundout))
        self.tts_say = partial(tts_actr.tts_session_say.remote, rgen_id=self.tts_sess)

class TestSession():
    id: UUID
    prompts: List[str]
    audio: torch.Tensor
    tts_sess_id: UUID
    stt_sess_id: UUID
    speaker_id: int
    nres: int = 0
    tot_error: float = 0.0
    def __init__(self, prompts, speaker_id, test_pipe, tokenizer, model):
        self.id = uuid4()
        self.prompts = list(prompts)
        self.speaker_id = speaker_id
        self.test_pipe = test_pipe
        self.tokenizer = tokenizer
        self.model = model
        self.audio = torch.zeros(0)

    def next_prompt(self, last_res=None):
        play_id = 0 if last_res is None else 1
        if len(self.prompts) > play_id:
            res = self.test_pipe.tts_say(text=self.prompts[play_id][0], speaker_id=self.speaker_id)
            ray.get(res)
        if play_id > 0:
            embedding1 = self.prompts.pop(0)[1]
            last_text, no_speech_prob = last_res
            embedding2 = get_embedding(self.tokenizer, self.model, last_text)
            similarity = 1 - cosine(embedding1, embedding2)
            self.nres += 1
            assert similarity >= 0.0 and similarity <= 1.0
            assert no_speech_prob >= 0.0 and no_speech_prob <= 1.0
            #self.tot_error += ((1.0 - similarity) + no_speech_prob) / 2
            self.tot_error += 1.0 - similarity
            print(f"Cosine Similarity[{self.speaker_id}]: average_error={self.average_error()}")
        return True if self.prompts else False

    def average_error(self):
        return self.tot_error / self.nres

@ray.remote
class InfernBenchActor():
    queue: Queue
    sessions: Dict[int, TestSession]
    def __init__(self, _): pass

    def loop(self):
        _prompts = [y for x in smith_set() + bender_set() + hal_set() for y in x.split('|')]
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        self.queue = Queue()
        self.sessions = {}
        tts_actrs = tuple(InfernTTSActor.remote() for _ in range(2))
        stt_actrs = tuple(InfernSTTActor.remote() for _ in range(2))
        for x in tuple(stt_actr.start.remote() for stt_actr in stt_actrs): ray.get(x)
        self_actor = ray.get_runtime_context().current_actor
        prompts = [(x, get_embedding(tokenizer, model, x)) for x in _prompts]
        def stt_actr(_s): return  stt_actrs[_s % len(stt_actrs)]
        def tts_actr(_s): return  tts_actrs[_s % len(tts_actrs)]
        res = []
        batch_size = 24 + 4
        target_nspeakers = 30
        test_pipes = list(TestPipe(tts_actr(i), stt_actr(i), self_actor.stt_done.remote)
                          for i in range(batch_size))
        draining = False
        class _next_ts():
            speaker_id = 6000
            def __init__(self, tokenizer, model):
                self.tokenizer = tokenizer
                self.model = model
            def __call__(self, tp, prompts):
                ts = TestSession(prompts[:1], self.speaker_id, tp, self.tokenizer, self.model)
                self.speaker_id += 1
                return ts
        next_ts = _next_ts(tokenizer, model)
        while True:
            if len(test_pipes) > 0 and not draining:
                tp = test_pipes.pop()
                try:
                    ts = next_ts(tp, prompts)
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
                    def __call__(self, tp, prompts):
                        ts = self.res.pop(0)
                        ts.prompts = prompts[:1]
                        ts.test_pipe = tp
                        return ts
                next_ts = _next_ts(res)
                res = []
                prompts.pop(0)
                draining = False

        for ts in res:
            print(f"Speaker[{ts.speaker_id}]: average_error={ts.average_error()}")

    def stt_done(self, session_id, text, no_speech_prob):
        ts = self.sessions[session_id]
        print(f'text: {text}, no_speech_prob: {no_speech_prob}')
        more = ts.next_prompt((text, no_speech_prob))
        if not more:
            del self.sessions[session_id]
            self.queue.put(ts)
        #if not self.sessions: self.stop()

    def stop(self):
        self.queue.put(None)
