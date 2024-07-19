from typing import Optional
from _thread import get_ident
from queue import Queue

import ray

#from Core.InfernConfig import InfernConfig
from SIP.InfernSIP import InfernSIP
from SIP.RemoteSession import RemoteSessionAccept, NewRemoteSessionRequest
from Cluster.InfernRTPActor import InfernRTPActor

@ray.remote(resources={"head": 0.5})
class InfernSIPActor():
    sip_stack: InfernSIP
    default_resources = {'head':1, 'stt': 1, 'tts':1, 'rtp': 1}
    def loop(self, inf_c:'InfernConfig'):
        #raise Exception("BP")
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        rtp_actr = self.rtp_actr = InfernRTPActor.options(max_concurrency=2).remote(inf_c.rtp_conf)
        sip_actr = ray.get_runtime_context().current_actor
        ray.get(rtp_actr.start.remote())
        self.sip_stack = InfernSIP(sip_actr, rtp_actr, inf_c)
        rtp_actr.loop.remote()
        rval = ED2.loop()
        ray.get(rtp_actr.stop.remote())
        return rval

    def new_sess(self, msg:NewRemoteSessionRequest):
        from sippy.Core.EventDispatcher import ED2
        rval = Queue()
        ED2.callFromThread(self.sip_stack.new_session, msg, rval)
        sip_sess, rtp_sess = rval.get()
        return (sip_sess.id, self.rtp_actr, rtp_sess.sess_id)

    def new_sess_accept(self, sip_sess_id, msg:RemoteSessionAccept):
        from sippy.Core.EventDispatcher import ED2
        sip_sess = self.sip_stack.get_session(sip_sess_id)
        rval = Queue()
        ED2.callFromThread(sip_sess.accept, msg, rval)
        rtp_sess = rval.get()
        return (self.rtp_actr, rtp_sess.sess_id)

    def new_sess_reject(self, sip_sess_id):
        from sippy.Core.EventDispatcher import ED2
        sip_sess = self.sip_stack.get_session(sip_sess_id)
        ED2.callFromThread(sip_sess.reject)

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
