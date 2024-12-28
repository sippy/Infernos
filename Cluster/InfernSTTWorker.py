from typing import Tuple, List
from os.path import expanduser, exists as path_exists
from subprocess import Popen, PIPE
from functools import partial

import ctranslate2
import transformers
from methodtools import lru_cache

import torch
from torch.nn import functional as F

from Cluster.STTSession import STTRequest, STTResult
from Cluster.InfernBatchedWorker import InfernBatchedWorker

class InfernSTTWorker(InfernBatchedWorker):
    max_batch_size: int = 4
    model: ctranslate2.models.Whisper
    processor: transformers.WhisperProcessor
    device: str
    cache_dir: str = '~/.cache/Infernos'
    sample_rate: int = 16000
    debug = False
    def __init__(self, device: str, model_name: str = "openai/whisper-large-v3"):
        super().__init__()
        if device != 'xpu':
            cache_dir = expanduser(f'{self.cache_dir}/{model_name}.ct2')
            if not any((path_exists(f'{cache_dir}/{_c}') for _c in ('model.bin', 'config.json', 'vocabulary.json'))):
                print(f'Converting "{model_name}" to "{cache_dir}"...')
                command = ['ct2-transformers-converter', '--model', model_name, '--output_dir', cache_dir]
                process = Popen(command, stdout=PIPE, stderr=PIPE)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise RuntimeError(f'{command[0]} failed with {process.returncode=}, {stdout=}, {stderr=}')
            self.model = ctranslate2.models.Whisper(cache_dir, device=device, compute_type="int8")
        else:
            from warnings import filterwarnings
            filterwarnings("ignore", category=FutureWarning)
            filterwarnings("ignore", category=UserWarning)
            from ipex_llm.transformers import AutoModelForSpeechSeq2Seq
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                load_in_4bit=True,
                torch_dtype="auto",
                device_map="auto",
                optimize_model=True,
                trust_remote_code=True,
                use_cache=True
            )
            self.model = model.to(device)
        self.processor = transformers.WhisperProcessor.from_pretrained(model_name)
        if device == 'xpu':
            self.no_speech_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|nospeech|>")
            self.process_audios = partial(self.processor, return_tensors="pt")
        else:
            self.process_audios = partial(self.processor, return_tensors="np")
        self.device = device
        self.infer_and_decode = partial(self.infer_and_decode_ct2 if device != 'xpu' else self.infer_and_decode_torch)

    def infer_and_decode_ct2(self, prompts, inputs):
        input_features = inputs.input_features
        features = ctranslate2.StorageView.from_array(input_features)
        try:
            results = self.model.generate(features, prompts, return_no_speech_prob=True)
        except RuntimeError as e:
            if 'out of memory' not in str(e) or len(prompts) == 1: raise
            torch.cuda.empty_cache()
            results = []
            for _if, _pr in zip(input_features, prompts):
                features = ctranslate2.StorageView.from_array([_if,])
                results.extend(self.model.generate(features, [_pr], return_no_speech_prob=True))
        decoded_results = ((self.processor.decode(r.sequences_ids[0]), r.no_speech_prob, r.sequences_ids[0])
                            for r in results)
        return decoded_results

    def infer_and_decode_torch(self, prompts, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        max_len = max(len(t) for t in prompts)
        prompts = torch.stack([
            F.pad(t, (0, max_len - t.size(0)), value=self.processor.tokenizer.pad_token_id)
            for t in (torch.tensor(pr, device=self.device) for pr in prompts)
        ])
        with torch.no_grad():
            forward_outputs = self.model(
                **inputs,
                decoder_input_ids=prompts,
            )
            logprobs = forward_outputs.logits[:, 0].log_softmax(-1)
            no_speech_probs = logprobs[:, self.no_speech_token_id].exp().tolist()
            gen_outputs = self.model.generate(
                **inputs,
                decoder_input_ids=prompts,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_seq = gen_outputs.sequences
        decoded_texts = self.processor.batch_decode(gen_seq, skip_special_tokens=True)
        decoded_results = (
            (text.strip(), nsp, gos.tolist()) for text, nsp, gos in
              zip(decoded_texts, no_speech_probs, gen_seq)
        )
        return decoded_results

    def process_batch(self, wis:List[Tuple[STTRequest, List[int]]]):
        if self.debug:
            print(f'InfernSTTWorker.process_batch: got {len(wis)=}')
        audios = [wi[0].chunk.audio for wi in wis]
        inputs = self.process_audios(audios, sampling_rate=self.sample_rate)
        prompts = self.get_prompt(tuple((wi[0].lang, wi[0].mode, wi[0].timestamps) for wi in wis))
        good_results = self.infer_and_decode(prompts, inputs)
        for (wi, c), (r, nsp, t) in zip(wis, good_results):
            # Remove leading and trailing space: "WhitespaceTokenizer adds a space at the beginning?" (copilot)
            if len(r) > 0 and r[0] == ' ': r = r[1:]
            if c is not None: c[:] = (c + t)[:-224]
            res = STTResult(text=r, no_speech_prob=nsp, req=wi)
            wi.text_cb(result = res)

    @lru_cache(maxsize=16)
    def get_prompt(self, options:Tuple[Tuple[str, str, bool]]):
        prompt = tuple(self.processor.tokenizer.convert_tokens_to_ids(
                [
                    "<|startoftranscript|>",
                   f"<|{language}|>",
                   f"<|{mode}|>",
                ] + ([] if timestamps else ["<|notimestamps|>"])
                ) for language, mode, timestamps in options)
        return prompt
