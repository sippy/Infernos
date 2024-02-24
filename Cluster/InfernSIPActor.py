from _thread import get_ident

import ray

from SIP.InfernSIP import InfernSIP
from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernRTPActor import InfernRTPActor

@ray.remote
class InfernSIPActor():
    def __init__(self, iao):
        self.iao = iao

    def loop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        rtp_actr = InfernRTPActor.options(max_concurrency=2).remote()
        tts_actr = InfernTTSActor.remote(rtp_actr)
        InfernSIP(tts_actr, self.iao)
        rtp_actr.loop.remote()
        rval = ED2.loop()
        ray.get(rtp_actr.stop.remote())
        return rval

    def stop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.callFromThread(ED2.breakLoop, 0)
