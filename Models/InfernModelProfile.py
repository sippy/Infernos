from typing import Optional, Tuple
from functools import partial

from .STT.Whisper import Whisper
from .STT.WhisperRT import WhisperRT

stt_models = {m.provides: m for m in (Whisper, WhisperRT)}

class InfernModelProfile():
    schema: dict = {
        'uses': { 'type': 'string' },
        'batch_size': {'type': 'integer', 'min': 1, 'max': 65535},
    }

    @staticmethod
    def getSchemaImpl(uses):
        if uses not in stt_models:
            raise ValueError(f'Unknown STT model "{uses}"')
        schema = stt_models[uses].schema.copy()
        schema.update(InfernModelProfile.schema)
        return schema
