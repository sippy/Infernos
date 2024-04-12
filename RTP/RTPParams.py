from typing import Tuple, Optional

class RTPParams():
    rtp_target: Tuple[str, int]
    out_ptime: int
    default_ptime: int = 20
    def __init__(self, rtp_target:Tuple[str, int], out_ptime:Optional[int]):
        assert isinstance(rtp_target, tuple) and len(rtp_target) == 2
        self.rtp_target = rtp_target
        self.out_ptime = out_ptime if out_ptime is not None else self.default_ptime
