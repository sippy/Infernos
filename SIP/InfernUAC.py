from sippy.CCEvents import CCEventTry, CCEventConnect
from sippy.SipCallId import SipCallId

from Cluster.RemoteRTPGen import RemoteRTPGen
from SIP.InfernUA import InfernUA, model_body
from RTP.RTPParams import RTPParams

class InfernUAC(InfernUA):
    def __init__(self, isip, cld):
        super().__init__(isip)
        call_id = SipCallId()
        body = model_body.getCopy()
        rtp_params = RTPParams((isip.sippy_c['nh_addr'][0], 0), None)
        self.rsess = RemoteRTPGen(isip.rtp_actr, rtp_params)
        self.stt_sess_id = isip.stt_actr.new_stt_session.remote(keep_context=True)
        print(f'{self.rsess.rtp_address=}')
        sect = body.content.sections[0]
        sect.c_header.addr, sect.m_header.port = self.rsess.rtp_address
        self.our_sdp_body = body
        event = CCEventTry((call_id, isip._o.cli, cld, body, None, "Dummy Joe"))
        self.username = isip._o.authname
        self.password = isip._o.authpass
        self.recvEvent(event)

    def outEvent(self, event, ua):
        if isinstance(event, CCEventConnect):
            code, reason, sdp_body = event.getData()
            rtp_params = self.extract_rtp_params(sdp_body)
            if rtp_params is None: return
            self.rsess.update(rtp_params)
            return
