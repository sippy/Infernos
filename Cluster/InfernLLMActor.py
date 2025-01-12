from typing import Dict
from uuid import UUID
from queue import Queue

import ray

from Cluster.InfernLLMWorker import InfernLLMWorker
from Cluster.LLMSession import LLMSession, LLMRequest, LLMInferRequest, LLMSessionParams

@ray.remote(num_gpus=1.0, resources={"llm": 1})
class InfernLLMActor():
    debug = True
    sessions: Dict[UUID, LLMSession]
    LLM: InfernLLMWorker

    def __init__(self):
        super().__init__()
        self.sessions = {}

    def start(self):
        for device in ('xpu', 'cuda', 'cpu'):
            try:
                self.llm = InfernLLMWorker(device)
            except (ValueError, RuntimeError):
                continue
            break
        else:
            raise RuntimeError('Failed to initialize LLM')
        self.llm.start()
        tq = Queue()
        def res_cb(result): tq.put(result)
        irs = tuple(LLMInferRequest(LLMRequest('What is your name?', None), [{}])
                    for _ in range(self.llm.max_batch_size))
        for _i in irs: _i.textout_cb = res_cb
        with self.llm.inf_queue.mutex:
            for ir in irs:
                self.llm.inf_queue.queue.append(ir)
            self.llm.inf_queue.not_empty.notify()
        for _ in irs:
            tq.get()

    def stop(self):
        self.llm.stop()

    def new_llm_session(self, sconf:LLMSessionParams):
        if self.debug: print('InfernLLMActor.new_llm_session')
        sess = LLMSession(self.llm, sconf)
        self.sessions[sess.id] = sess
        return sess.id

    def llm_session_end(self, sess_id):
        if self.debug: print('InfernLLMActor.llm_session_end')
        sess = self.sessions[sess_id]
        sess.stop()
        del self.sessions[sess_id]

    def llm_session_textin(self, sess_id, req:LLMRequest):
        if self.debug: print('InfernLLMActor.llm_session_textin')
        sess = self.sessions[sess_id]
        sess.textin(req)
        return sess_id

    def llm_session_context_add(self, sess_id, content:str, role:str = 'user'):
        if self.debug: print('InfernLLMActor.llm_session_context_add')
        sess = self.sessions[sess_id]
        sess.context_add(content, role)
        return sess_id
