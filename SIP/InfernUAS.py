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

import ray

from sippy.UA import UA
from sippy.CCEvents import CCEventTry, CCEventConnect, CCEventFail, CCEventUpdate
from sippy.MsgBody import MsgBody
from sippy.SdpOrigin import SdpOrigin
from sippy.SipConf import SipConf
from sippy.SdpMedia import MTAudio
from sippy.SipReason import SipReason

from Cluster.RemoteRTPGen import RemoteRTPGen, RTPGenError
from Cluster.RemoteTTSSession import RemoteTTSSession, TTSSessionError

ULAW_PT = 0
ULAW_RM = 'PCMU/8000'
body_txt = 'v=0\r\n' + \
  'o=- 380960 380960 IN IP4 192.168.22.95\r\n' + \
  's=-\r\n' + \
  'c=IN IP4 192.168.22.95\r\n' + \
  't=0 0\r\n' + \
 f'm=audio 16474 RTP/AVP {ULAW_PT}\r\n' + \
 f'a=rtpmap:0 {ULAW_RM}\r\n' + \
  'a=ptime:30\r\n' + \
  'a=sendrecv\r\n' + \
  '\r\n'
model_body = MsgBody(body_txt)
model_body.parse()

class InfernUASConf(object):
    cli = 'infernos_uas'
    cld = 'infernos_uac'
    authname = None
    authpass = None
    nh_addr = ('192.168.0.102', 5060)
    laddr = None
    lport = None
    logger = None

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

class CCEventSentDone: pass
class CCEventSTTTextIn: pass

class InfernTTSUAS(UA):
    debug = True
    id: UUID
    rsess: RemoteRTPGen
    our_sdp_body: MsgBody
    _tsess: RemoteTTSSession = None
    prompts = None
    autoplay = True

    def __init__(self, sippy_c, sip_actr, tts_actr, stt_actr, rtp_actr, req, sip_t, prompts):
        self.id = uuid4()
        self.sip_actr, self.rtp_actr = sip_actr, rtp_actr
        self._tsess = RemoteTTSSession(tts_actr)
        activate_cb = partial(sip_actr.sess_event.remote, sip_sess_id=self.id, event=CCEventSTTTextIn())
        self.stt_sess_id = stt_actr.new_stt_session.remote(self._tsess.sess_id, activate_cb)
        super().__init__(sippy_c, self.outEvent)
        assert sip_t.noack_cb is None
        sip_t.noack_cb = self.sess_term
        self.prompts = prompts
        self.recvRequest(req, sip_t)

    def extract_rtp_target(self, sdp_body):
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
        return (sect.c_header.addr, sect.m_header.port)

    def outEvent(self, event, ua):
        if isinstance(event, CCEventUpdate):
            sdp_body = event.getData()
            rtp_target = self.extract_rtp_target(sdp_body)
            if rtp_target is None: return
            self.rsess.update(rtp_target)
            self.send_uas_resp()
            return
        if not isinstance(event, CCEventTry):
            #ua.disconnect()
            return
        cId, cli, cld, sdp_body, auth, caller_name = event.getData()
        rtp_target = self.extract_rtp_target(sdp_body)
        if rtp_target is None: return
        try:
            self.rsess = RemoteRTPGen(self.rtp_actr, ray.get(self.stt_sess_id), rtp_target)
            self._tsess.start(self.rsess.get_soundout())
        except RTPGenError as e:
            event = InfernUASFailure(code=500, reason=str(e))
            self.recvEvent(event)
            raise e
        self.disc_cbs = (self.sess_term,)
        body = model_body.getCopy()
        sect = body.content.sections[0]
        sect.c_header.addr, sect.m_header.port = self.rsess.rtp_address
        self.our_sdp_body = body
        self.send_uas_resp()
        self.recvEvent(CCEventSentDone())

    def recvEvent(self, event):
        if self._tsess is None: return
        if isinstance(event, CCEventSTTTextIn):
            r = event.kwargs['text']
            if r.strip() == "Let's talk.":
                self.autoplay = False
            if self.autoplay:
                return
            self._tsess.say(r)
            return
        if isinstance(event, CCEventSentDone):
            if not self.autoplay:
                return
            if len(self.prompts) == 0:
                self.sess_term()
                return
            next_sentence_cb = partial(self.sip_actr.sess_event.remote, sip_sess_id=self.id, event=event)
            self._tsess.say(self.prompts.pop(0), next_sentence_cb)
            return
        super().recvEvent(event)

    def send_uas_resp(self):
        self.our_sdp_body.content.o_header = SdpOrigin()
        oevent = CCEventConnect((200, 'OK', self.our_sdp_body.getCopy()))
        return self.recvEvent(oevent)

    def sess_term(self, ua=None, rtime=None, origin=None, result=0):
        print('disconnected')
        if self._tsess is None:
            return
        self._tsess.end()
        self._tsess = None
        self.rsess.join()
        if ua != self:
            self.disconnect()

    def __del__(self):
        if self.debug:
            print('InfernUAS.__del__')
