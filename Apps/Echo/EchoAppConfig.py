from .EchoProfile import EchoProfile


class EchoAppConfig:
    schema = {
        'echo': {'type': 'dict', 'schema': EchoProfile.schema},
    }
