from .LTProfile import LTProfile

class LTAppConfig():
    schema: dict = {
        'live_translator': {
            'type': 'dict',
            'schema': {
                **LTProfile.schema,
            }
        },
        'live_translator_precache': {'type': 'boolean'},
    }
