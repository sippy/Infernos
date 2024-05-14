from typing import Optional

from sippy.CCEvents import CCEventTry, CCEventConnect
from sippy.SipCallId import SipCallId
from sippy.SdpMediaDescription import a_header

from Cluster.RemoteRTPGen import RemoteRTPGen
from SIP.InfernUA import InfernUA, model_body
from RTP.RTPParams import RTPParams
from Core.Codecs.G711 import G711Codec
from Core.Codecs.G722 import G722Codec
from .RemoteSession import NewRemoteSessionRequest
from .InfernUAS import InfernUAS
from .InfernSIPProfile import InfernSIPProfile

class InfernUAC(InfernUA):
    uas:Optional[InfernUAS]=None
    offer=(G711Codec, G722Codec)
    def __init__(self, isip, msg:NewRemoteSessionRequest):
        sip_prof: InfernSIPProfile = msg.sip_prof
        if msg.conn_sip_sess_id is not None:
            self.uas = isip.get_session(msg.conn_sip_sess_id)
        super().__init__(isip, nh_address = sip_prof.nh_addr)
        if msg.disc_cb is not None:
            self.disc_cbs += (msg.disc_cb,)
        call_id = SipCallId()
        body = model_body.getCopy()
        rtp_params = RTPParams((sip_prof.nh_addr[0], 0), None)
        rtp_params.codec = self.offer[0]
        self.rsess = RemoteRTPGen(isip.rtp_actr, rtp_params)
        print(f'{self.rsess.rtp_address=}')
        sect = body.content.sections[0]
        sect.c_header.addr, sect.m_header.port = self.rsess.rtp_address
        sect.a_headers.insert(0, a_header(f'ptime:{rtp_params.out_ptime}'))
        for i, codec in enumerate(self.offer):
            sect.a_headers.insert(i, a_header(codec.rtpmap()))
        self.our_sdp_body = body
        event = CCEventTry((call_id, sip_prof.cli, msg.cld, body, None, "Dummy Joe"))
        self.username = sip_prof.authname
        self.password = sip_prof.authpass
        self.disc_cbs = (self.sess_term,)
        self.recvEvent(event)

    def outEvent(self, event, ua):
        if isinstance(event, CCEventConnect):
            code, reason, sdp_body = event.getData()
            rtp_params = self.extract_rtp_params(sdp_body)
            if rtp_params is None: return
            self.rsess.update(rtp_params)
        if self.uas is not None:
            self.uas.recvEvent(event)
