#try: import intel_extension_for_pytorch as ipex
#except ModuleNotFoundError: ipex = None

from typing import Dict, Union, Optional
from functools import partial
from uuid import UUID
from _thread import get_ident

from ray import ray

from config.InfernGlobals import InfernGlobals as IG
from Core.AudioChunk import AudioChunk
from RTP.InfernRTPIngest import InfernRTPIngest
from RTP.InfernRTPEPoint import InfernRTPEPoint
from Core.AStreamMarkers import ASMarkerGeneric
from RTP.AudioInput import AudioInput
from RTP.RTPParams import RTPParams

@ray.remote(num_gpus=0.01, resources={"rtp": 1})
class InfernRTPActor():
    devices = ('cuda', 'cpu')
    device: str
    sessions: Dict[UUID, InfernRTPEPoint]
    ring: InfernRTPIngest
    def __init__(self):
        self.sessions = {}

    def new_rtp_session(self, rtp_params:RTPParams):
        print(f'{IG.stdtss()}: new_rtp_session')
        rep = InfernRTPEPoint(rtp_params, self.ring, self._get_direct_soundout)
        self.sessions[rep.id] = rep
        return (rep.id, rep.rserv.uopts.laddress)

    def rtp_session_connect(self, rtp_id, ain:AudioInput):
        print(f'{IG.stdtss()}: rtp_session_connect[{str(rtp_id)[:6]}]')
        rep = self.sessions[rtp_id]
        rep.connect(ain)

    def rtp_session_end(self, rtp_id):
        print(f'{IG.stdtss()}: rtp_session_end')
        rep = self.sessions[rtp_id]
        rep.writer.end()

    def rtp_session_soundout(self, rtp_id, chunk:Union[AudioChunk, ASMarkerGeneric]):
        rep = self.sessions[rtp_id]
        return rep.soundout(chunk)

    def _get_direct_soundout(self, rtp_id):
        rep = self.sessions[rtp_id]
        return rep.soundout

    def rtp_session_join(self, rtp_id):
        print(f'{IG.stdtss()}: rtp_session_join')
        rep = self.sessions[rtp_id]
        rep.shutdown()
        del self.sessions[rtp_id]

    def rtp_session_update(self, rtp_id, rtp_params:RTPParams):
        print(f'{IG.stdtss()}: rtp_session_update')
        rep = self.sessions[rtp_id]
        rep.update(rtp_params)

    def start(self):
        for device in self.devices:
            self.ring = InfernRTPIngest(device)
            try:
                self.ring.start()
            except AssertionError:
                print(f'{device} did not worl')
                continue
            self.device = device
            break
        else:
            raise RuntimeError('No suitable device found')

    def loop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.my_ident = get_ident()
        rval = ED2.loop()
        self.ring.stop()
        self.ring.join()
        return rval

    def stop(self):
        from sippy.Core.EventDispatcher import ED2
        ED2.callFromThread(ED2.breakLoop, 0)
