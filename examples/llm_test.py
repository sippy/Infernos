import ray
from sys import stderr
from time import monotonic
from uuid import UUID
from functools import partial
from time import sleep
from Cluster.InfernLLMActor import InfernLLMActor
from Cluster.LLMSession import LLMRequest

#@ray.remote(resources={"head": 1})
#class text_in(result):

class TimedLLMRequest(LLMRequest):
    queue_ts: float
    proc_start_ts: float
    def __init__(self, text:str, lms:UUID, lma:InfernLLMActor):
        tin = partial(self.text_in, lms=lms, lma=lma)
        super().__init__(text, tin)
        self.queue_ts = monotonic()

    def _proc_start_cb(self):
        self.proc_start_ts = monotonic()

    def text_in(self, result:str, lms:UUID, lma:InfernLLMActor):
        from sys import stderr as _stderr
        itime = monotonic() - self.proc_start_ts
        print(f'text_in: got {result=}, inference time: {itime}', file=_stderr)
        req = TimedLLMRequest('Hello, can I speak to the CEO?', lms, lma)
        lma.llm_session_textin.remote(lms,  req)


ray.init(num_gpus=2, resources = {'llm':1,'head':1})

print('Initializing InfernLLMActor...', file=stderr)
llm_actor = InfernLLMActor.remote()
ray.get(llm_actor.start.remote())
print('InfernLLMActor is ready', file=stderr)


flms = [llm_actor.new_llm_session.remote() for _ in range(100)]
print(f'Created {len(flms)} sessions', file=stderr)
def sess(lms):
    req = TimedLLMRequest('<Incoming call from "Doe Joe" +11233742223>', lms, llm_actor)
    return llm_actor.llm_session_textin.remote(lms,  req)
futs = [sess(lms) for lms in flms]
for f in futs:
    ray.get(f)
sleep(3600)
