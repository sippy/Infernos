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

from functools import partial

import ray

from sippy.CCEvents import CCEventTry

from Cluster.RemoteRTPGen import RemoteRTPGen, RTPGenError
from Cluster.RemoteTTSSession import RemoteTTSSession
from Cluster.STTSession import STTRequest
from SIP.InfernUA import InfernUA, model_body, InfernUASFailure

class CCEventSentDone: pass
class CCEventSTTTextIn: pass

class InfernTTSUAS(InfernUA):
    lang: str
    prompts = None
    autoplay = True
    _tsess: RemoteTTSSession = None
    def __init__(self, sippy_c, sip_actr, tts_actr, stt_actr, rtp_actr, lang,
                 prompts, req, sip_t):
        super().__init__(sippy_c, sip_actr, rtp_actr)
        self._tsess = RemoteTTSSession(tts_actr)
        self.stt_actr, self.stt_sess_id = stt_actr, stt_actr.new_stt_session.remote()
        self.lang = lang
        assert sip_t.noack_cb is None
        sip_t.noack_cb = self.sess_term
        self.prompts = prompts
        self.recvRequest(req, sip_t)

    class STTProxy():
        debug = True
        stt_do: callable
        stt_done: callable
        def __init__(self, stt_actr, stt_sess_id, lang, stt_done):
            self.stt_do = partial(stt_actr.stt_session_soundin.remote, sess_id=stt_sess_id)
            self.lang, self.stt_done = lang, stt_done

        def __call__(self, chunk):
            if self.debug:
                print(f'STTProxy.chunk_in {len(chunk)=}')
            sreq = STTRequest(chunk, self.stt_done, self.lang)
            self.stt_do(req=sreq)

    def outEvent(self, event, ua):
        if not isinstance(event, CCEventTry):
            super().outEvent(event, ua)
            return
        cId, cli, cld, sdp_body, auth, caller_name = event.getData()
        rtp_target = self.extract_rtp_target(sdp_body)
        if rtp_target is None: return
        self.stt_sess_id = ray.get(self.stt_sess_id)
        text_cb = partial(self.sip_actr.sess_event.remote, sip_sess_id=self.id, event=CCEventSTTTextIn())
        vad_handler = self.STTProxy(self.stt_actr, self.stt_sess_id, self.lang, text_cb)
        try:
            self.rsess = RemoteRTPGen(self.rtp_actr, vad_handler, rtp_target)
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
            res = event.kwargs['result']
            nsp = res.no_speech_prob
            if nsp > 0.3: return
            print(f'STT: -> "{res.text} {res.no_speech_prob}"')
            if res.text.strip() == "Let's talk.":
                self.autoplay = False
            if self.autoplay:
                return
            self._tsess.say(res.text)
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

    def sess_term(self, ua=None, rtime=None, origin=None, result=0):
        print('disconnected')
        if self._tsess is None:
            return
        self._tsess.end()
        self.stt_actr.stt_session_end.remote(sess_id=self.stt_sess_id)
        self._tsess = None
        super().sess_term(ua=ua, rtime=rtime, origin=origin, result=result)
