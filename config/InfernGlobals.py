from safetorch.InfernTorcher import InfernTorcher

class InfernGlobals():
    _instance = None
    torcher: InfernTorcher

    def __init__(self):
        self.torcher = InfernTorcher()

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance