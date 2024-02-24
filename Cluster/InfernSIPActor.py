from _thread import get_ident

import ray

from SIP.InfernSIP import InfernSIP

@ray.remote
class InfernSIPActor():
    def __init__(self, iao):
        self.iao = iao

    def stop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.callFromThread(ED2.breakLoop, 0)

    def loop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        sip_stack = InfernSIP(self.iao)
        return (sip_stack.loop())