from typing import Optional, Tuple
from queue import Queue, Empty as QueueEmpty
from os.path import expanduser, exists as path_exists
from subprocess import Popen, PIPE

import ctranslate2
import transformers
from methodtools import lru_cache

from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun
from Cluster.STTSession import STTRequest, STTResult

class InfernSTTWorker(InfernWrkThread):
    max_batch_size: int = 4
    model: ctranslate2.models.Whisper
    processor: transformers.WhisperProcessor
    device: str
    cache_dir: str = '~/.cache/Infernos'
    inf_queue: Queue[Optional[STTRequest]]
    sample_rate: int = 16000
    def __init__(self, device: str, model_name: str = "openai/whisper-large-v3"):
        super().__init__()
        cache_dir = expanduser(f'{self.cache_dir}/{model_name}.ct2')
        if not any((path_exists(f'{cache_dir}/{_c}') for _c in ('model.bin', 'config.json', 'vocabulary.json'))):
            print(f'Converting "{model_name}" to "{cache_dir}"...')
            command = ['ct2-transformers-converter', '--model', model_name, '--output_dir', cache_dir]
            process = Popen(command, stdout=PIPE, stderr=PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(f'{command[0]} failed with {process.returncode=}, {stdout=}, {stderr=}')
        self.model = ctranslate2.models.Whisper(cache_dir, device=device, compute_type="int8")
        self.processor = transformers.WhisperProcessor.from_pretrained(model_name)
        self.inf_queue = Queue()
        self.device = device

    def infer(self, wi:STTRequest):
        self.inf_queue.put(wi)

    #idx: int = 0

    def next_batch(self) -> STTRequest:
        wis = []
        while len(wis) < self.max_batch_size:
            if len(wis) == 0:
                wi = self.inf_queue.get()
            else:
                try: wi = self.inf_queue.get_nowait()
                except QueueEmpty: break
            if wi is None:
                return None
            #sf.write(f'/tmp/wi{self.idx}.wav', wi.audio, samplerate=self.sample_rate)
            #self.idx += 1
            wis.append(wi)
        return wis

    def run(self):
        super().thread_started()
        while self.get_state() == RTPWrkTRun:
            wis = self.next_batch()
            if wis is None:
                break
            print(f'InfernSTTWorker.run: got {wis} from inf_queue')
            audios = [wi.audio for wi in wis]
            inputs = self.processor(audios, return_tensors="np", sampling_rate=self.sample_rate)
            features = ctranslate2.StorageView.from_array(inputs.input_features)
            #ldet = dict([(i, features[i]) for i, wi in enumerate(wis) if wi.stt_sess.lang is None])
            #results = self.model.detect_language(features)
            #print(f'{results=}')
            ##prompt = [self.processor.tokenizer.convert_tokens_to_ids(
            ##    [
            ##        "<|startoftranscript|>",
            ##       f"<|{language}|>",
            ##        "<|transcribe|>",
            ##        "<|notimestamps|>",  # Remove this token to generate timestamps.
            ##    ]) for language in (wi.lang for wi in wis)]
            prompt = self.get_prompt(tuple(wi.lang for wi in wis))
            results = self.model.generate(features, prompt, return_no_speech_prob=True)
            good_results = [(wi, self.processor.decode(r.sequences_ids[0]), r.no_speech_prob)
                             for wi, r in zip(wis, results)]
            for wi, r, nsp in good_results:
                wi.text_cb(result = STTResult(text=r, no_speech_prob=nsp))
            #with torch.no_grad():
            #    audio = wi.audio.to(self.device)
            #    res = wi.stt_sess.model(audio)
            #    wi.res_queue.put(res)

    @lru_cache(maxsize=16)
    def get_prompt(self, langs:Tuple[str]):
        prompt = tuple(self.processor.tokenizer.convert_tokens_to_ids(
                [
                    "<|startoftranscript|>",
                   f"<|{language}|>",
                    "<|transcribe|>",
                    "<|notimestamps|>",  # Remove this token to generate timestamps.
                ]) for language in langs)
        return prompt

    def stop(self):
        self.inf_queue.put(None)
        super().stop()
