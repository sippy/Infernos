class GenCodec():
    srate:int = 8000 # sample rate
    crate:int = 8000 # clock rate
    ptype:int        # payload type
    ename:str        # encoding name

    def __init__(self):
        assert self.ptype is not None and self.ename is not None

    @classmethod
    def rtpmap(cls):
        assert all(hasattr(cls, attr) for attr in ('ptype', 'ename'))
        return f'rtpmap:{cls.ptype} {cls.ename}/{cls.crate}'
