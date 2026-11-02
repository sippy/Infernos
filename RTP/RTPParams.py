from typing import Optional, Tuple

from sippy.Rtp.Params import RTPParams as SippyRTPParams
from sippy.Rtp.Codecs.G711 import G711Codec
from sippy.Rtp.Codecs.G722 import G722Codec


class RTPParams(SippyRTPParams):
    """Infernos defaults with sippy media parameters and real codec classes."""

    codec = G711Codec

    def __init__(self, rtp_target: Optional[Tuple[str, int]], out_ptime: Optional[int] = None,
                 out_sr: int = SippyRTPParams.default_sr, rtp_proto: Optional[str] = None):
        if rtp_proto is None:
            rtp_proto = 'IP6' if rtp_target is not None and ':' in rtp_target[0] else 'IP4'
        super().__init__(rtp_target,
                         out_ptime=self.default_ptime if out_ptime is None else out_ptime,
                         out_sr=out_sr, rtp_proto=rtp_proto)
