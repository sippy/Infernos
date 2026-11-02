import ray

from .EchoActor import EchoActor


class EchoProfile:
    schema = {
        'profiles': {
            'type': 'dict',
            'required': True,
            'keysrules': {'type': 'string'},
            'valuesrules': {
                'type': 'dict',
                'schema': {
                    'stt_lang': {'type': 'string', 'empty': False},
                    'tts_lang': {'type': 'string', 'empty': False},
                    'greeting': {'type': 'string'},
                },
            },
        },
    }

    def __init__(self, name, conf):
        self.name = name
        self.stt_lang = conf.get('stt_lang', 'en')
        self.tts_lang = conf.get('tts_lang', 'en')
        self.greeting = conf.get('greeting')
        self.actor = None

    def finalize(self, iconf):
        pass

    def getActor(self, iconf, sip_act):
        if self.actor is None:
            actor = EchoActor.remote()
            ray.get(actor.start.remote(self, sip_act))
            self.actor = actor
        return self.actor
