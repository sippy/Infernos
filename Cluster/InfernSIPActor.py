from _thread import get_ident
from queue import Queue

import ray

from SIP.InfernSIP import InfernSIP
from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernSTTActor import InfernSTTActor
from Cluster.InfernRTPActor import InfernRTPActor

@ray.remote(resources={"head": 1})
class InfernSIPActor():
    sip_stack: InfernSIP
    default_resources = {'head':1, 'stt': 1, 'tts':1, 'rtp': 1}
    def __init__(self, iao):
        self.iao = iao

    def loop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        tts_langs = ('it', 'en')
        stt_langs = ('ru', 'it')
        stt_actr = InfernSTTActor.remote()
        rtp_actr = InfernRTPActor.options(max_concurrency=2).remote()
        sip_actr = ray.get_runtime_context().current_actor
        tts_actrs = dict((l, InfernTTSActor.remote()) for l in tts_langs)
        for fut in tuple(_a.start.remote(**_k) for _a, _k in ((stt_actr, {}), (rtp_actr, {})) +
                         tuple((a, {'lang':l, 'output_sr':8000}) for l, a in tts_actrs.items())):
            ray.get(fut)
        self.sip_stack = InfernSIP(sip_actr, tts_actrs, stt_actr, rtp_actr, self.iao,
                                   tts_langs, stt_langs)
        rtp_actr.loop.remote()
        rval = ED2.loop()
        ray.get(rtp_actr.stop.remote())
        ray.get(stt_actr.stop.remote())
        return rval

    def new_sess(self, cld:str):
        from sippy.Core.EventDispatcher import ED2
        rval = Queue()
        ED2.callFromThread(self.sip_stack.new_session, cld, rval)
        sip_sess = rval.get()
        return sip_sess.id

    def sess_term(self, sip_sess_id):
        from sippy.Core.EventDispatcher import ED2
        sip_sess = self.sip_stack.get_session(sip_sess_id)
        ED2.callFromThread(sip_sess.sess_term)

    def sess_event(self, sip_sess_id, event, **kwargs):
        from sippy.Core.EventDispatcher import ED2
        sip_sess = self.sip_stack.get_session(sip_sess_id)
        event.kwargs = kwargs
        ED2.callFromThread(sip_sess.recvEvent, event)

    def stop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.callFromThread(ED2.breakLoop, 0)
