from _thread import get_ident

import ray

from SIP.InfernSIP import InfernSIP
from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernSTTActor import InfernSTTActor
from Cluster.InfernRTPActor import InfernRTPActor

@ray.remote
class InfernSIPActor():
    sip_stack: InfernSIP
    def __init__(self, iao):
        self.iao = iao

    def loop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        stt_actr = InfernSTTActor.remote()
        stt_actr.start.remote()
        rtp_actr = InfernRTPActor.options(max_concurrency=2).remote(stt_actr)
        sip_actr = ray.get_runtime_context().current_actor
        tts_actr = InfernTTSActor.remote(rtp_actr, sip_actr)
        self.sip_stack = InfernSIP(tts_actr, self.iao)
        rtp_actr.loop.remote()
        rval = ED2.loop()
        ray.get(rtp_actr.stop.remote())
        ray.get(stt_actr.stop.remote())
        return rval

    def sess_term(self, sip_sess_id):
        from sippy.Core.EventDispatcher import ED2
        sip_sess = self.sip_stack.get_session(sip_sess_id)
        ED2.callFromThread(sip_sess.sess_term)

    def stop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.callFromThread(ED2.breakLoop, 0)
