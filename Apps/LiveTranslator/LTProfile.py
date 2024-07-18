from typing import Tuple, Optional

import ray

from Cluster.InfernSIPActor import InfernSIPActor

from .LTActor import LTActor

class LTProfile():
    name: str
    tts_langs: Tuple[str]
    stt_langs: Tuple[str]
    _outbound_spec: str
    outbound_conn: 'InfernSIPProfile'
    outbount_params: str
    actor: Optional[LTActor] = None
    precache: bool

    def __init__(self, name, conf, precache):
        self.name = name
        self.tts_langs = tuple(conf['tts_langs'])
        self.stt_langs = tuple(conf['stt_langs'])
        if not precache:
            self._outbound = conf['outbound']
        self.precache = precache

    def finalize(self, iconf:'InfernConfig'):
        if not self.precache:
            sip_cname, params = self._outbound.split(';', 1)
            self.outbound_conn = iconf.connectors[sip_cname]
            self.outbount_params = params
        else:
            actor = LTActor.remote()
            res = ray.get(actor.precache.remote(self))

    def getActor(self, iconf:'InfernConfig', sip_act:InfernSIPActor):
        if self.actor is None:
            self.actor = LTActor.remote()
            ray.get(self.actor.start.remote(self, sip_act))
        return self.actor
