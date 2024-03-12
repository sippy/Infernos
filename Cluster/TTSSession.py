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

from typing import Optional, Union, List
from time import monotonic
from uuid import uuid4, UUID
from queue import Queue

from TTSRTPOutput import TTSSMarkerEnd, TTSSMarkerNewSent
from Cluster.RemoteRTPGen import RemoteRTPGenFromId
from Core.InfernWrkThread import InfernWrkThread, RTPWrkTStop, RTPWrkTInit

class TTSSMarkerNewSentCB(TTSSMarkerNewSent):
    def __init__(self, tts_actr, tts_sess: 'TTSSession'):
        super().__init__()
        self.tts_actr = tts_actr
        self.tts_sess_id = tts_sess.id

    def on_proc(self, tro_self):
        print(f'{monotonic():4.3f}: TTSSMarkerNewSentCB.on_proc')
        self.tts_actr.tts_session_next_sentence.remote(self.tts_sess_id)

class TTSRequest():
    text: str
    speaker = None
    final: bool
    def __init__(self, text: str, speaker, final: bool = False):
        self.text = text
        self.speaker = speaker
        self.final = final

class TTSSession(InfernWrkThread):
    debug = True
    id: UUID
    tts = None
    ptime = 0.030
    worker: RemoteRTPGenFromId
    eos_m: TTSSMarkerNewSentCB
    next_sentence_q: Queue[TTSRequest]
    autoplay = True

    def __init__(self, tts, sess_term):
        super().__init__()
        self.id = uuid4()
        self.tts = tts
        self.sess_term = sess_term

    def start(self, tts_actr, rtp_actr, rtp_sess_id, text):
        worker = RemoteRTPGenFromId(rtp_actr, rtp_sess_id)
        self.state_lock.acquire()
        assert self.get_state(locked=True) == RTPWrkTInit
        self.worker = worker
        self.text = [text,] if type(text) == str else text
        self.eos_m = TTSSMarkerNewSentCB(tts_actr, self)
        self.next_sentence_q = Queue()
        self.state_lock.release()
        super().start()
        # XXX kick-in the worker
        self.next_sentence()

    def run(self):
        super().thread_started()
        disconnected = False
        while self.get_state() != RTPWrkTStop:
            sent = self.next_sentence_q.get()
            if sent is None: break
            sents = sent.text.split('|')
            for i, p in enumerate(sents):
                if self.get_state() == RTPWrkTStop: break
                print(f'{monotonic():4.3f}: Playing', p)
                self.tts.tts_rt(p, self.worker.soundout, sent.speaker)
                if not sent.final or i < len(sents) - 1:
                    self.worker.soundout(self.eos_m)
            if sent.final:
                disconnected = True
                break
        if self.get_state() == RTPWrkTStop:
            self.worker.end()

        self.worker.soundout(TTSSMarkerEnd())
        self.worker.join()
        if not disconnected:
            self.sess_term()
        self.sess_term()
        del self.sess_term
        del self.worker

    def next_sentence(self):
        if not self.autoplay: return
        print(f'{monotonic():4.3f}: TTSSession.next_sentence')
        speaker = self.tts.get_rand_voice()
        sent = self.text.pop(0)
        final = (len(self.text) == 0)
        self.next_sentence_q.put(TTSRequest(sent, speaker, final))

    def stopintro(self):
        print(f'{monotonic():4.3f}: TTSSession.stopintro')
        self.autoplay = False

    def stop(self):
        self.next_sentence_q.put(None)
        super().stop()

    def __del__(self):
        if self.debug:
            print('TTSSession.__del__')
