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
from functools import partial, lru_cache

import ray

from sippy.CCEvents import CCEventTry

from Cluster.RemoteRTPGen import RemoteRTPGen, RTPGenError
from Cluster.RemoteTTSSession import RemoteTTSSession
from Cluster.STTSession import STTRequest
from SIP.InfernUA import InfernUA, model_body, InfernUASFailure
from RTP.RTPOutputWorker import TTSSMarkerNewSent
from Core.AudioChunk import AudioChunk
from Core.T2T.Translator import Translator

class CCEventSentDone: pass
class CCEventSTTTextIn: pass
class CCEventVADIn: pass

import pickle
import gzip
from random import choice

@lru_cache(maxsize=4)
def get_top_speakers(lang:str):
    skips = 0
    i = 0
    res = []
    while True:
        try:
            with gzip.open(f'checkpoint/{lang}/speaker.{i}.{lang}.pkl.gz', 'rb') as file:
                res.append(pickle.load(file))
        except FileNotFoundError:
            skips += 1
            if skips > 200: break
        i += 1
    if len(res) == 0:
        return None
    gen = max(r.nres for r in res)
    return tuple(r.speaker_id for r in res if r.nres == gen)

class InfernTTSUAS(InfernUA):
    stt_lang: str
    prompts = None
    autoplay = True
    _tsess: RemoteTTSSession = None
    translator: Optional[Translator] = None
    def __init__(self, isip, req, sip_t):
        super().__init__(isip.sippy_c, isip.sip_actr, isip.rtp_actr)
        self._tsess = RemoteTTSSession(isip.tts_actr)
        self.stt_actr, self.stt_sess_id = isip.stt_actr, isip.stt_actr.new_stt_session.remote()
        self.stt_lang = isip.stt_lang
        assert sip_t.noack_cb is None
        sip_t.noack_cb = self.sess_term
        self.prompts = isip.getPrompts()
        self.speakers = get_top_speakers(isip.tts_lang)
        if isip.tts_lang != isip.stt_lang:
            self.translator = Translator(isip.stt_lang, isip.tts_lang)
        self.recvRequest(req, sip_t)

    class STTProxy():
        debug = True
        stt_do: callable
        stt_done: callable
        vad_mirror: callable
        def __init__(self, uas, stt_done, vad_mirror):
            self.stt_do = partial(uas.stt_actr.stt_session_soundin.remote, sess_id=uas.stt_sess_id)
            self.lang, self.stt_done = uas.stt_lang, stt_done
            self.vad_mirror = vad_mirror

        def __call__(self, chunk:AudioChunk):
            if self.debug:
                print(f'STTProxy.chunk_in {len(chunk.audio)=}')
            sreq = STTRequest(chunk.audio.numpy(), self.stt_done, self.lang)
            self.stt_do(req=sreq)
            self.vad_mirror(chunk=chunk)

    def outEvent(self, event, ua):
        if not isinstance(event, CCEventTry):
            super().outEvent(event, ua)
            return
        cId, cli, cld, sdp_body, auth, caller_name = event.getData()
        rtp_target = self.extract_rtp_target(sdp_body)
        if rtp_target is None: return
        self.stt_sess_id = ray.get(self.stt_sess_id)
        text_cb = partial(self.sip_actr.sess_event.remote, sip_sess_id=self.id, event=CCEventSTTTextIn())
        vad_cb = partial(self.sip_actr.sess_event.remote, sip_sess_id=self.id, event=CCEventVADIn())
        vad_handler = self.STTProxy(self, text_cb, vad_cb)
        try:
            self.rsess = RemoteRTPGen(self.rtp_actr, vad_handler, rtp_target)
            rtp_soundout = self.rsess.get_soundout()
            self._tsess.start(rtp_soundout)
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
            text = res.text if  self.translator is None else self.translator.translate(res.text)
            if any(t.strip() == "Let's talk." for t in (res.text, text)):
                self.autoplay = False
            if self.autoplay:
                return
            speaker_id=None if self.speakers is None else choice(self.speakers)
            self._tsess.say(text, speaker_id=speaker_id)
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
        if isinstance(event, CCEventVADIn):
            chunk = event.kwargs['chunk']
            assert isinstance(chunk, AudioChunk)
            chunk.track_id += 1
            print(f'VAD: {chunk.track_id=}')
            self.rsess.soundout(chunk=chunk)
            self.rsess.soundout(TTSSMarkerNewSent(track_id=chunk.track_id))
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
