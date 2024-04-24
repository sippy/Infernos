from time import monotonic

import ray

from Core.Exceptions.InfernSessNotFoundErr import InfernSessNotFoundErr

class ASMarkerGeneric():
    track_id: int
    debug: bool = False
    def __init__(self, track_id:int=0):
        self.track_id = track_id

class ASMarkerNewSent(ASMarkerGeneric):
    # This runs in the context of the RTPOutputWorker thread
    def on_proc(self, tro_self, *args): pass

class ASMarkerSentDoneCB(ASMarkerNewSent):
    def __init__(self, done_cb:callable, sync:bool=False, **kwargs):
        super().__init__(**kwargs)
        self.done_cb = done_cb
        self.sync = sync

    def on_proc(self, tro_self):
        print(f'{monotonic():4.3f}: ASMarkerSentDoneCB.on_proc')
        x = self.done_cb()
        if self.sync:
            try: ray.get(x)
            except InfernSessNotFoundErr: pass
