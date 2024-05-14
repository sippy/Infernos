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

    def __init__(self, name, conf):
        self.name = name
        self.tts_langs = tuple(conf['tts_langs'])
        self.stt_langs = tuple(conf['stt_langs'])
        self._outbound = conf['outbound']

    def finalize(self, iconf:'InfernConfig'):
        sip_cname, params = self._outbound.split(';', 1)
        self.outbound_conn = iconf.connectors[sip_cname]
        self.outbount_params = params

    def getActor(self, iconf:'InfernConfig', sip_act:InfernSIPActor):
        if self.actor is None:
            self.actor = LTActor.remote()
            ray.get(self.actor.start.remote(self, sip_act))
        return self.actor
