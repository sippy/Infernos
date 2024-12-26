import ray
from typing import Optional

from Cluster.InfernSIPActor import InfernSIPActor

from .AIAActor import AIAActor


class AIAProfile():
    schema: dict = {
        'profiles': {
            'type': 'dict',
            'keysrules': {'type': 'string'},
            'valuesrules': {
                'type': 'dict',
                'schema': {
                    'tts_lang': {'type': 'string'},
                    'stt_lang': {'type': 'string'},
                }
            }
        }
    }
    stt_lang: str = 'en'
    tts_lang: str = 'en'
    actor: Optional[AIAActor] = None

    def __init__(self, name, conf):
        self.name = name
        self.tts_lang = conf['tts_lang']
        self.stt_lang = conf['stt_lang']

    def finalize(self, iconf:'InfernConfig'):
        pass

    def getActor(self, iconf:'InfernConfig', sip_act:InfernSIPActor):
        if self.actor is None:
            self.actor = AIAActor.remote()
            ray.get(self.actor.start.remote(self, sip_act))
        return self.actor
