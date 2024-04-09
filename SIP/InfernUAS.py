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

from typing import Optional, List, Tuple
from functools import partial, lru_cache

import ray

from sippy.CCEvents import CCEventTry

from config.InfernGlobals import InfernGlobals as IG
from Cluster.RemoteRTPGen import RemoteRTPGen, RTPGenError
from Cluster.RemoteTTSSession import RemoteTTSSession
from Cluster.STTSession import STTRequest
from SIP.InfernUA import InfernUA, model_body, InfernUASFailure
from RTP.AudioInput import AudioInput
from Core.AudioChunk import AudioChunk, AudioChunkFromURL
from Core.T2T.Translator import Translator
from RTP.RTPOutputWorker import TTSSMarkerEnd

class CCEventSentDone: pass
class CCEventSTTTextIn:
    def __init__(self, direction):
        self.direction = direction

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

class STTProxy():
    debug = True
    stt_do: callable
    stt_done: callable
    vad_mirror: callable
    def __init__(self, uas, lang, stt_done, vad_mirror, stt_sess_id, direction):
        self.stt_do = partial(uas.stt_actr.stt_session_soundin.remote, sess_id=stt_sess_id)
        self.lang, self.stt_done = lang, stt_done
        self.vad_mirror = vad_mirror
        self.eng = uas.vds.eng
        self.direction = direction

    # This method runs in the context of the RTP Actor
    def __call__(self, chunk:AudioChunk):
        if self.debug:
            dir = 'A' if self.direction == 0 else 'B'
            print(f'STTProxy: VAD({dir}): {len(chunk.audio)=} {chunk.track_id=}')
        self.vad_mirror(chunk=self.eng)
        sreq = STTRequest(chunk.audio.numpy(), self.stt_done, self.lang)
        self.stt_do(req=sreq)

class TTSProxy():
    tts_consume: callable
    def __init__(self, tts_consume):
        self.tts_consume = tts_consume

    def __call__(self, chunk:AudioChunk):
        chunk.track_id = 1
        self.tts_consume(chunk=chunk)

class SessionInfo():
    soundout: callable
    rsess_pause: callable
    rsess_connect: callable
    translator: callable
    get_speaker: callable
    tts_say: callable
    def __init__(self, uas:'InfernTTSUAS', xua:InfernUA, yua:InfernUA, direction):
        text_cb = partial(uas.sip_actr.sess_event.remote, sip_sess_id=uas.id, event=CCEventSTTTextIn(direction))
        self.soundout = xua.rsess.get_soundout()
        vad_cb = self.soundout
        vad_handler = STTProxy(uas, uas.stt_lang[direction], text_cb, vad_cb, xua.stt_sess_id, direction)
        self.rsess_pause = partial(uas.rtp_actr.rtp_session_connect.remote, xua.rsess.sess_id,
                                   AudioInput(vad_chunk_in=vad_handler))
        ysoundout = yua.rsess.get_soundout()
        self.rsess_connect = partial(uas.rtp_actr.rtp_session_connect.remote, xua.rsess.sess_id,
                                     AudioInput(ysoundout, vad_handler))
        self.translator = (lambda x: x) if uas.translators[direction] is None else uas.translators[direction].translate
        self.get_speaker = (lambda: None) if uas.speakers[direction] is None else partial(choice, uas.speakers[direction])
        self.tts_say = partial(uas._tsess[direction].say, done_cb=self.rsess_connect)
        self.tts_soundout = TTSProxy(ysoundout)

class Sessions():
    info: Tuple[SessionInfo]
    def __init__(self, uas:'InfernTTSUAS'):
        self.info = (
            SessionInfo(uas, uas, uas.bob_sess[0], 0),
            SessionInfo(uas, uas.bob_sess[0], uas, 1),
            )
        uas._tsess[0].start(self.info[1].soundout)
        uas._tsess[1].start(self.info[0].soundout)
        for i, t in zip(self.info, uas._tsess):
            i.rsess_connect()
            t.start(i.tts_soundout)

class InfernTTSUAS(InfernUA):
    stt_lang: str
    prompts = None
    autoplay = False
    _tsess: List[RemoteTTSSession]
    translators: List[Optional[Translator]]
    def __init__(self, isip, req, sip_t):
        super().__init__(isip)
        self.vds = self.VADSignals()
        self._tsess = [RemoteTTSSession(isip.tts_actr[lang]) for lang in isip.tts_lang]
        self.stt_actr, self.stt_sess_id = isip.stt_actr, isip.stt_actr.new_stt_session.remote(keep_context=True)
        self.stt_lang = isip.stt_lang
        assert sip_t.noack_cb is None
        sip_t.noack_cb = self.sess_term
        self.prompts = isip.getPrompts()
        self.speakers = [get_top_speakers(l) for l in isip.tts_lang]
        self.translators = [None if l1 == l2 else IG.get_translator(l1, l2) for l1, l2 in zip(isip.stt_lang, isip.tts_lang)]
        self.bob_sess = isip.new_session('16047861714')
        self.recvRequest(req, sip_t)

    class VADSignals():
        def __init__(self):
            eng, deng= [AudioChunkFromURL(f'https://github.com/commaai/openpilot/blob/master/selfdrive/assets/sounds/{n}.wav?raw=true') for n in ('engage', 'disengage')]
            eng.track_id = 1
            self.eng = ray.put(eng)
            self.deng = ray.put(deng)

    overlay_id = 42
    #vds = VADSignals()

    def outEvent(self, event, ua):
        if not isinstance(event, CCEventTry):
            super().outEvent(event, ua)
            return
        cId, cli, cld, sdp_body, auth, caller_name = event.getData()
        rtp_target = self.extract_rtp_target(sdp_body)
        if rtp_target is None: return
        self.stt_sess_id = ray.get(self.stt_sess_id)

        try:
            self.rsess = RemoteRTPGen(self.rtp_actr, rtp_target)
            self.fabric = Sessions(self)
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
            sinfo = self.fabric.info[event.direction]
            sdir = 'A->B' if event.direction == 0 else 'B->A'
            print(f'STT: {sdir} "{res.text=}" {res.no_speech_prob=}')
            text = sinfo.translator(res.text)
            if event.direction == 0 and any(t.strip() == "Let's talk." for t in (res.text, text)):
                self.autoplay = False
            if self.autoplay:
                return
            speaker_id = sinfo.get_speaker()
            sinfo.rsess_pause()
            print(f'TTS: {sdir} "{text=}" {speaker_id=}')
            sinfo.tts_say(text, speaker_id=speaker_id)
            return
        if isinstance(event, CCEventSentDone):
            if not self.autoplay:
                return
            if len(self.prompts) == 0:
                self.sess_term()
                return
            next_sentence_cb = partial(self.sip_actr.sess_event.remote, sip_sess_id=self.id, event=event)
            self._tsess[0].say(self.prompts.pop(0), next_sentence_cb)
            return
        super().recvEvent(event)

    def sess_term(self, ua=None, rtime=None, origin=None, result=0):
        print('disconnected')
        if self._tsess is None:
            return
        for ts in self._tsess: ts.end()
        self.stt_actr.stt_session_end.remote(sess_id=self.stt_sess_id)
        self.stt_actr.stt_session_end.remote(sess_id=self.bob_sess[0].stt_sess_id)
        self._tsess = None
        super().sess_term(ua=ua, rtime=rtime, origin=origin, result=result)
        self.bob_sess[0].sess_term(ua=ua, rtime=rtime, origin=origin, result=result)
