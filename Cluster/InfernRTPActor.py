#try: import intel_extension_for_pytorch as ipex
#except ModuleNotFoundError: ipex = None

from typing import Dict, Union, List
from uuid import UUID
from _thread import get_ident

from ray import ray

from sippy.Network_server import RTP_port_allocator

from config.InfernGlobals import InfernGlobals as IG
from Core.AudioChunk import AudioChunk
from Core.AStreamMarkers import ASMarkerGeneric
from Core.Exceptions.InfernSessNotFoundErr import InfernSessNotFoundErr
from RTP.InfernRTPIngest import InfernRTPIngest
from RTP.InfernRTPEPoint import InfernRTPEPoint
from RTP.AudioInput import AudioInput
from RTP.RTPParams import RTPParams
from RTP.InfernRTPConf import InfernRTPConf

class RTPSessNotFoundErr(InfernSessNotFoundErr): pass

@ray.remote(num_gpus=0.01, resources={"rtp": 1})
class InfernRTPActor():
    devices = ('mps', 'cuda', 'cpu')
    device: str
    sessions: Dict[UUID, InfernRTPEPoint]
    thumbstones: List[UUID]
    ring: InfernRTPIngest
    palloc: RTP_port_allocator
    inf_rc: InfernRTPConf
    def __init__(self, inf_rc:InfernRTPConf):
        self.sessions = {}
        self.thumbstones = []
        self.inf_rc = inf_rc

    def new_rtp_session(self, rtp_params:RTPParams):
        print(f'{IG.stdtss()}: new_rtp_session')
        rep = InfernRTPEPoint(self.inf_rc, rtp_params, self.ring, self._get_direct_soundout)
        self.sessions[rep.id] = rep
        return (rep.id, rep.rserv.uopts.laddress)

    def rtp_session_connect(self, rtp_id, ain:AudioInput):
        print(f'{IG.stdtss()}: rtp_session_connect[{str(rtp_id)[:6]}]')
        rep = self._get_session(rtp_id)
        rep.connect(ain)

    def rtp_session_end(self, rtp_id, relaxed:bool=False):
        print(f'{IG.stdtss()}: rtp_session_end')
        try:
            rep = self._get_session(rtp_id)
        except RTPSessNotFoundErr:
            if relaxed or rtp_id in self.thumbstones: return
            raise
        rep.writer.end()

    def rtp_session_soundout(self, rtp_id, chunk:Union[AudioChunk, ASMarkerGeneric]):
        try:
            rep = self._get_session(rtp_id)
        except RTPSessNotFoundErr:
            if rtp_id in self.thumbstones:
                return
            raise
        return rep.soundout(chunk)

    def _get_direct_soundout(self, rtp_id):
        rep = self._get_session(rtp_id)
        return rep.soundout

    def rtp_session_join(self, rtp_id):
        print(f'{IG.stdtss()}: rtp_session_join')
        rep = self._get_session(rtp_id)
        rep.shutdown()
        del self.sessions[rtp_id]
        self.thumbstones.append(rtp_id)
        if len(self.thumbstones) > 100:
            self.thumbstones = self.thumbstones[-100:]

    def rtp_session_update(self, rtp_id, rtp_params:RTPParams):
        print(f'{IG.stdtss()}: rtp_session_update')
        rep = self._get_session(rtp_id)
        rep.update(rtp_params)

    def start(self):
        for device in self.devices:
            self.ring = InfernRTPIngest(device)
            try:
                self.ring.start()
            except (AssertionError, RuntimeError):
                print(f'{device} did not work')
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

    def _get_session(self, rtp_id:UUID) -> InfernRTPEPoint:
        try: return self.sessions[rtp_id]
        except KeyError: raise RTPSessNotFoundErr(f'No RTP session found for {rtp_id}')
