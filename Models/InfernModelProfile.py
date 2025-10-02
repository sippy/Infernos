from typing import Dict, Optional

from .STT.Whisper import Whisper
from .STT.WhisperRT import WhisperRT

from cerberus import Validator

from Core.ConfigValidators import InfernConfigParseErr

stt_models = {m.provides: m for m in (Whisper, WhisperRT)}

class InfernModelProfile():
    schema: dict = {
        'uses': {'type': 'string', 'required': True},
        'batch_size': {'type': 'integer', 'min': 1, 'max': 65535},
        'parameters': {
            'type': 'dict',
            'default_setter': lambda _: {},
            'allow_unknown': True,
        },
    }

    name: str
    uses: str
    batch_size: Optional[int]
    parameters: Dict[str, object]
    model_cls: type

    def __init__(self, name: str, conf: dict):
        self.name = name
        self.uses = conf['uses']
        if self.uses not in stt_models:
            raise InfernConfigParseErr(f'Unknown STT model "{self.uses}" for profile "{name}"')
        self.model_cls = stt_models[self.uses]
        params = dict(conf.get('parameters') or {})
        validator = Validator(self.model_cls.schema, allow_unknown=False)
        if not validator.validate(params):
            raise InfernConfigParseErr(
                f'Invalid parameters for STT model profile "{name}" ({self.uses}): {validator.errors}'
            )
        self.parameters = validator.document
        self.batch_size = conf.get('batch_size')
