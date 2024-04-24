from safetorch.InfernTorcher import InfernTorcher
from threading import Lock
from functools import lru_cache
from time import monotonic

import torchaudio.transforms as T

from Core.T2T.Translator import Translator

class InfernGlobals():
    _lock = Lock()
    _instance = None
    torcher: InfernTorcher

    @lru_cache
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InfernGlobals, cls).__new__(cls)
                cls.torcher = InfernTorcher()
        return cls._instance

    @staticmethod
    @lru_cache(maxsize=8)
    def get_resampler(from_sr:int, to_sr:int, device:str='cpu'):
        return T.Resample(orig_freq=from_sr, new_freq=to_sr).to(device)

    @staticmethod
    @lru_cache(maxsize=8)
    def get_translator(from_lang:str, to_lang:str, **kwa):
        return Translator(from_lang, to_lang, **kwa)

    def stdtss():
        return f'{monotonic():4.3f}'
