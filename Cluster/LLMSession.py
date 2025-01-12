from typing import List, Tuple, Optional
from time import monotonic
from functools import partial
from uuid import uuid4, UUID

class LLMRequest():
    id: UUID
    text: str
    textout_cb: callable
    auto_ctx_add: bool = True
    def __init__(self, text:str, textout_cb:callable):
        self.text, self.textout_cb = text, textout_cb
        self.id = uuid4()

class LLMResult():
    req_id: UUID
    text: str
    def __init__(self, text:str, req_id:UUID):
        self.text, self.req_id = text, req_id

class LLMInferRequest():
    req: LLMRequest
    context: Tuple[dict]
    textout_cb: callable

    def __init__(self, req:LLMRequest, context:List[dict]):
        self.req, self.context = req, tuple(context)

class LLMSessionParams():
    system_prompt: str
    def __init__(self, system_prompt:str):
        self.system_prompt = system_prompt

class LLMSession():
    id: UUID
    context: List[dict]
    debug: bool = False
    def __init__(self, llm:'InfernLLMWorker', params:LLMSessionParams):
        self.id = uuid4()
        self.context = [{"role": "system", "content": params.system_prompt}]
        self.llm = llm
        
    def context_add(self, content:str, role:str = "user"):
        if self.debug:
            print(f'{monotonic():4.3f}: LLMSession.context_add: {self.context=}, {content=}')
        if len(self.context) > 0 and self.context[-1]["role"] == role:
            self.context[-1]["content"] += f' {content}'
        else:
            self.context.append({"role": role, "content": content})

    def textin(self, req:LLMRequest):
        if self.debug:
            print(f'{monotonic():4.3f}: LLMSession.textin: ${req.text=}, {req.textout_cb=} {self.context=}')
        self.context_add(req.text)
        ireq = LLMInferRequest(req, self.context)
        if hasattr(req, '_proc_start_cb'):
            ireq._proc_start_cb = req._proc_start_cb
        ireq.textout_cb = partial(self.textout, req = req)
        self.llm.infer(ireq)

    def textout(self, req:LLMRequest, result:LLMResult):
        if self.debug:
            print(f'{monotonic():4.3f}: LLMSession.textout: {result.text=}')
        if req.auto_ctx_add:
            self.context_add(result.text, "assistant")
        req.textout_cb(result = result)

    def stop(self):
        if self.debug: print('STTSession.stop')
        del self.llm
