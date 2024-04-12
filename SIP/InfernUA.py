# Copyright (c) 2018 Sippy Software, Inc. All rights reserved.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from uuid import uuid4, UUID
from functools import partial

from sippy.UA import UA
from sippy.CCEvents import CCEventFail, CCEventUpdate, CCEventConnect
from sippy.MsgBody import MsgBody
from sippy.SdpOrigin import SdpOrigin
from sippy.SipConf import SipConf
from sippy.SdpMedia import MTAudio
from sippy.SipReason import SipReason

from Cluster.RemoteRTPGen import RemoteRTPGen
from RTP.RTPParams import RTPParams

ULAW_PT = 0
ULAW_RM = 'PCMU/8000'
ULAW_PTIME = RTPParams.default_ptime
body_txt = 'v=0\r\n' + \
  'o=- 380960 380960 IN IP4 192.168.22.95\r\n' + \
  's=-\r\n' + \
  'c=IN IP4 192.168.22.95\r\n' + \
  't=0 0\r\n' + \
 f'm=audio 16474 RTP/AVP {ULAW_PT}\r\n' + \
 f'a=rtpmap:0 {ULAW_RM}\r\n' + \
 f'a=ptime:{ULAW_PTIME}\r\n' + \
  'a=sendrecv\r\n' + \
  '\r\n'
model_body = MsgBody(body_txt)
model_body.parse()

class InfernUAConf(object):
    cli = 'infernos_uas'
    cld = 'infernos_uac'
    authname = None
    authpass = None
    nh_addr = ('192.168.0.102', 5060)
    laddr = None
    lport = None
    logger = None
    new_sess_offer: callable = None

    def __init__(self):
        self.laddr = SipConf.my_address
        self.lport = SipConf.my_port

class InfernUASFailure(CCEventFail):
    default_code = 488
    _code_msg = {default_code : 'Not Acceptable Here',
                 500          : 'Server Internal Error'}
    def __init__(self, reason=None, code=default_code):
        self.code, self.msg = code, self._code_msg[code]
        super().__init__((self.code, self.msg))
        self.reason = SipReason(protocol='SIP', cause=self.code,
                                reason=reason)

class InfernUA(UA):
    debug = True
    id: UUID
    rsess: RemoteRTPGen
    our_sdp_body: MsgBody

    def __init__(self, isip):
        self.id = uuid4()
        self.sip_actr, self.rtp_actr = isip.sip_actr, isip.rtp_actr
        super().__init__(isip.sippy_c, self.outEvent, nh_address=isip.sippy_c['nh_addr'])

    def extract_rtp_target(self, sdp_body):
        p = self.extract_rtp_params(sdp_body)
        if p is None: return None
        return p.rtp_target

    def extract_rtp_params(self, sdp_body):
        if sdp_body == None:
            event = InfernUASFailure("late offer/answer is not supported at this time, sorry")
            self.recvEvent(event)
            return
        sdp_body.parse()
        sects = [s for s in sdp_body.content.sections
                 if s.m_header.type == MTAudio and
                 ULAW_PT in s.m_header.formats]
        if len(sects) == 0:
            event = InfernUASFailure("only G.711u audio is supported at this time, sorry")
            self.recvEvent(event)
            return None
        sect = sects[0]
        try:
            ptime = int(next(x for x in sect.a_headers if x.name == 'ptime').value)
        except StopIteration:
            ptime = None
        return RTPParams((sect.c_header.addr, sect.m_header.port), ptime)

    def outEvent(self, event, ua):
        if isinstance(event, CCEventUpdate):
            sdp_body = event.getData()
            rtp_params = self.extract_rtp_params(sdp_body)
            if rtp_params is None: return
            self.rsess.update(rtp_params)
            self.send_uas_resp()
            return

    def send_uas_resp(self):
        self.our_sdp_body.content.o_header = SdpOrigin()
        oevent = CCEventConnect((200, 'OK', self.our_sdp_body.getCopy()))
        return super().recvEvent(oevent)

    def sess_term(self, ua=None, rtime=None, origin=None, result=0):
        print('disconnected')
        if self.rsess is None:
            return
        self.rsess.end()
        self.rsess.join()
        if ua != self:
            self.disconnect()
        self.rsess = None

    def __del__(self):
        if self.debug:
            print('InfernUA.__del__')
