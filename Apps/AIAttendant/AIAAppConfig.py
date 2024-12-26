from .AIAProfile import AIAProfile

class AIAAppConfig():
    schema: dict = {
        'ai_attendant': {
            'type': 'dict',
            'schema': {
                **AIAProfile.schema,
            }
        },
    }
