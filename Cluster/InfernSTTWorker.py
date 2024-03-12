from typing import Optional
from queue import Queue, Empty as QueueEmpty

import torch
import ctranslate2
import transformers
import soundfile as sf

from Core.InfernWrkThread import InfernWrkThread, RTPWrkTRun
from Cluster.STTSession import STTSession

class STTWI():
    stt_sess: STTSession
    audio: torch.Tensor
    res_queue: Queue

class InfernSTTWorker(InfernWrkThread):
    max_batch_size: int = 4
    model: ctranslate2.models.Whisper
    processor: transformers.WhisperProcessor
    device: str
    inf_queue: Queue[Optional[STTWI]]
    sample_rate: int = 16000
    def __init__(self, device: str):
        super().__init__()
        self.model = ctranslate2.models.Whisper("whisper-large-v3.ct2", device=device, compute_type="int8")
        self.processor = transformers.WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.inf_queue = Queue()
        self.device = device

    def infer(self, stt_sess, audio, res_queue):
        wi = STTWI()
        wi.stt_sess, wi.audio, wi.res_queue = stt_sess, audio, res_queue
        self.inf_queue.put(wi)

    idx: int = 0

    def next_batch(self):
        wis = []
        while len(wis) < self.max_batch_size:
            if len(wis) == 0:
                wi = self.inf_queue.get()
            else:
                try: wi = self.inf_queue.get_nowait()
                except QueueEmpty: break
            if wi is None:
                return None
            sf.write(f'/tmp/wi{self.idx}.wav', wi.audio, samplerate=self.sample_rate)
            self.idx += 1
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
            prompt = [self.processor.tokenizer.convert_tokens_to_ids(
                [
                    "<|startoftranscript|>",
                   f"<|{language}|>",
                    "<|translate|>",
                    "<|notimestamps|>",  # Remove this token to generate timestamps.
                ]) for language in (wi.stt_sess.lang for wi in wis)]
            results = self.model.generate(features, prompt, return_no_speech_prob=True)
            print(f'{results=}')
            good_results = [self.processor.decode(r.sequences_ids[0]) for r in results if r.no_speech_prob <= 0.3]
            for r in good_results: print(r)
            if any(r.strip() == "Let's talk." for r in good_results):
                print('BINGO')
            #with torch.no_grad():
            #    audio = wi.audio.to(self.device)
            #    res = wi.stt_sess.model(audio)
            #    wi.res_queue.put(res)

    def stop(self):
        self.inf_queue.put(None)
        super().stop()

