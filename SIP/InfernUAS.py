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

from typing import Optional
from uuid import uuid4, UUID
from queue import Queue

from sippy.CCEvents import CCEventTry, CCEventConnect
from sippy.SdpMediaDescription import a_header

from Cluster.RemoteRTPGen import RemoteRTPGen, RTPGenError
from SIP.InfernUA import InfernUA, model_body, InfernUASFailure
from SIP.RemoteSession import RemoteSessionAccept
from SIP.InfernSIPProfile import InfernSIPProfile
from SIP.SipSessInfo import SipSessInfo
from Core.Codecs.G711 import G711Codec
from Core.Codecs.G722 import G722Codec

class CCEventSentDone: pass
class CCEventSTTTextIn:
    def __init__(self, direction):
        self.direction = direction

class InfernUAS(InfernUA):
    rsess: Optional[RemoteRTPGen] = None
    etry: Optional[CCEventTry] = None
    auto_answer: bool
    accept_codecs = (G722Codec, G711Codec)
    def __init__(self, isip, req, sip_t, auto_answer=True):
        super().__init__(isip)
        assert sip_t.noack_cb is None
        self.auto_answer = auto_answer
        sip_t.noack_cb = self.sess_term
#        self.prompts = isip.getPrompts()
        self.recvRequest(req, sip_t)

    def outEvent(self, event, ua):
        if not isinstance(event, CCEventTry):
            super().outEvent(event, ua)
            return
        self.etry = event
        cId, cli, cld, sdp_body, auth, caller_name = event.getData()
        rtp_params = self.extract_rtp_params(sdp_body, accept=self.accept_codecs)
        if rtp_params is None:
            event = InfernUASFailure(code=500)
            self.recvEvent(event)
            return

        try:
            self.rsess = RemoteRTPGen(self.rtp_actr, rtp_params)
        except RTPGenError as e:
            event = InfernUASFailure(code=500, reason=str(e))
            self.recvEvent(event)
            raise e
        self.disc_cbs = (self.sess_term,)
        body = model_body.getCopy()
        sect = body.content.sections[0]
        sect.c_header.addr, sect.m_header.port = self.rsess.rtp_address
        sect.a_headers.insert(0, a_header(f'ptime:{rtp_params.out_ptime}'))
        sect.a_headers.insert(0, a_header(rtp_params.codec.rtpmap()))
        sect.m_header.formats = [rtp_params.codec.ptype,]
        self.our_sdp_body = body
        if self.auto_answer:
            self.send_uas_resp()

    def recvEvent(self, event):
        if not self.auto_answer and isinstance(event, CCEventConnect):
            return self.send_uas_resp()
        super().recvEvent(event)

class InfernLazyUAS(InfernUAS):
    id: UUID
    def __init__(self, sip_stack:'InfernSIP', sip_prof:InfernSIPProfile, req, sip_t):
        self._id = self.id = uuid4()
        self._sip_stack = sip_stack
        self._sip_prof = sip_prof
        self._req = req
        self._sip_t = sip_t
        sip_t.cancel_cb = self.cancelled
        resp = req.genResponse(100, 'Trying')
        sip_stack.sippy_c['_sip_tm'].sendResponse(resp)

    def accept(self, rsa:RemoteSessionAccept, rval:Queue):
        self._sip_t.cancel_cb = None
        super().__init__(self._sip_stack, self._req, self._sip_t, rsa.auto_answer)
        self.id = self._id
        del self._sip_stack, self._req, self._sip_t, self._id
        if rsa.disc_cb is not None:
            self.disc_cbs += (rsa.disc_cb,)
        rval.put(self.rsess)

    def reject(self):
        resp = self._req.genResponse(666, 'OOPS')
        self._sip_stack.sippy_c['_sip_tm'].sendResponse(resp)
        del self._sip_stack, self._req, self._sip_t, self._id

    def cancelled(self, *args):
        del self._sip_stack, self._req, self._sip_t, self._id

    def get_session_info(self) -> SipSessInfo:
        call_id = str(self._req.getHFBody('call-id'))
        from_hf = self._req.getHFBody('from')
        from_name = from_hf.getUri().name
        from_number = from_hf.getUrl().username
        return SipSessInfo(call_id, from_number, from_name)
