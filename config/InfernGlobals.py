from safetorch.InfernTorcher import InfernTorcher
from threading import Lock
from functools import lru_cache

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
