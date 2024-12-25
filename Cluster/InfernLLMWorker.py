from typing import Tuple, List, Iterator
from os.path import exists as path_exists
from itertools import chain
from functools import partial

import torch
import torch.nn.functional as F

from transformers import TextStreamer

from Cluster.InfernBatchedWorker import InfernBatchedWorker
from Cluster.InfernTTSWorker import get_torch_hw
from Cluster.LLMSession import LLMResult, LLMInferRequest

class ResultsStreamer(TextStreamer):
    debug = False
    sync_on = ('. ', '? ', '! ', '\n')
    decode_batch_size = 8
    def __init__(self, wis:List[LLMInferRequest], upper:'InfernLLMWorker'):
        super().__init__(tokenizer=upper.llm_tokenizer)
        self.wi_cbs = tuple(wi.textout_cb for wi in wis)
        self.newLLMResult = tuple(partial(LLMResult, req_id=wi.req.id) for wi in wis)
        batch_size = len(wis)
        self.oposs = [0 for _ in range(batch_size)]
        self.current_tokens = None
        self.batch_decode = partial(upper.llm_tokenizer.batch_decode, skip_special_tokens=True)

    def put(self, token_ids):
        if self.current_tokens is None:
            self.current_tokens = torch.zeros((token_ids.shape[0], 0), dtype=torch.long)
            return
        if token_ids.dim() == 1:  # Shape [batch_size]
            token_ids = token_ids.unsqueeze(1)
        self.current_tokens = torch.cat([self.current_tokens, token_ids], dim=1)
        if self.current_tokens.shape[1] % self.decode_batch_size == 0:
            return
        results = self.batch_decode(self.current_tokens)
        for (ir, r), op, cb, newLR in zip(enumerate(results), self.oposs, self.wi_cbs, self.newLLMResult):
            new_content = r[op:]
            if len(new_content) == 0: continue
            sp = (op + pos + len(c) for c in self.sync_on if (pos:=new_content.rfind(c)) >= 0)
            try:
                spos = next(sp)
            except StopIteration:
                continue
            r = r[op:spos-1]
            if len(r) < 10: continue
            cb(result=newLR(r))
            self.oposs[ir] = spos
        if self.debug:
            print(f'{self.oposs=} {self.current_tokens.shape=}')

    def end(self):
        if self.debug:
            print(f'finished: {self.current_tokens.shape=}')
        results = self.batch_decode(self.current_tokens)
        for r, op, cb, newLR in zip(results, self.oposs, self.wi_cbs, self.newLLMResult):
            if len(r) == op: continue
            cb(result=newLR(r[op:]))
        del self.current_tokens
        del self.wi_cbs

class InfernLLMWorker(InfernBatchedWorker):
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    model_cache_dir = f"/tmp/saved_model.{model_name}"
    max_batch_size: int = 8
    debug = True
    llm_model: object
    llm_tokenizer: object
    output_sr: int

    def __init__(self, device=None):
        from warnings import filterwarnings
        filterwarnings("ignore", category=FutureWarning)
        filterwarnings("ignore", category=UserWarning)
        from transformers import AutoTokenizer
        from ipex_llm.transformers import AutoModelForCausalLM
        super().__init__()
        if device is None:
            device = get_torch_hw()
        def load_model(mn):
            m = AutoModelForCausalLM.from_pretrained(mn, torch_dtype="auto",
                    device_map="auto",
                    optimize_model=True,
                    trust_remote_code=True,
                    load_in_4bit=True,
                    use_cache=True
                )
            if mn != self.model_cache_dir:
                m.save_low_bit(self.model_cache_dir)
            return m.to(device)
        if path_exists(self.model_cache_dir):
            try:
                model = AutoModelForCausalLM.load_low_bit(self.model_cache_dir,
                                                          trust_remote_code=True)
            except Exception:
                model = load_model(self.model_name)
        else:
            model = load_model(self.model_name)
        self.llm_model = model.to(device)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
 
    def process_batch(self, wis:List[LLMInferRequest]):
        if self.debug:
            print(f'InfernLLMWorker.process_batch: got {len(wis)=}')
        streamer = ResultsStreamer(wis, self)
        with torch.no_grad():
            messages = [self.llm_tokenizer.apply_chat_template(list(r.context), tokenize=False,
                        add_generation_prompt=True)
                        for r  in wis]
            model_inputs = self.llm_tokenizer(messages, return_tensors="pt", padding=True).to(self.llm_model.device)
            self.llm_model.generate(
                **model_inputs,
                max_new_tokens=16 * 1024,
                output_scores=True,
                return_dict_in_generate=True,
                streamer=streamer,
            )
            torch.xpu.synchronize()
