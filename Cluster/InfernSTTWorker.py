from typing import Tuple, List
from os.path import expanduser, exists as path_exists
from subprocess import Popen, PIPE
from fractions import Fraction

import ctranslate2
import transformers
from methodtools import lru_cache

import torch

from Cluster.STTSession import STTRequest, STTResult
from Cluster.InfernBatchedWorker import InfernBatchedWorker

class InfernSTTWorker(InfernBatchedWorker):
    max_batch_size: int = 4
    model: ctranslate2.models.Whisper
    processor: transformers.WhisperProcessor
    device: str
    cache_dir: str = '~/.cache/Infernos'
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
        self.device = device

    def process_batch(self, wis:List[STTRequest]):
        print(f'InfernSTTWorker.process_batch: got {len(wis)=}')
        audios = [wi.audio for wi in wis]
        inputs = self.processor(audios, return_tensors="np", sampling_rate=self.sample_rate)
        features = ctranslate2.StorageView.from_array(inputs.input_features)
        prompt = self.get_prompt(tuple(wi.lang for wi in wis))
        try:
            results = self.model.generate(features, prompt, return_no_speech_prob=True)
        except RuntimeError as e:
            if 'out of memory' not in str(e) or len(wis) == 1: raise
            torch.cuda.empty_cache()
            results = []
            for i in range(len(wis)):
                features = ctranslate2.StorageView.from_array(inputs.input_features[i:i+1])
                results.extend(self.model.generate(features, prompt[i:i+1], return_no_speech_prob=True))
        print(f'{results[0].sequences_ids[0][0]=}')
        good_results = [(wi, self.processor.decode(r.sequences_ids[0], skip_special_tokens=True), r.no_speech_prob)
                            for wi, r in zip(wis, results)]
        for wi, r, nsp in good_results:
            duration = Fraction(len(wi.audio), self.sample_rate)
            wi.text_cb(result = STTResult(text=r, no_speech_prob=nsp, duration=duration))

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