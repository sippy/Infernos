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

import ray

from TTSRTPOutput import TTSSMarkerEnd, TTSSMarkerNewSent
from Cluster.RemoteRTPGen import RemoteRTPGenFromId
from Core.InfernWrkThread import InfernWrkThread, RTPWrkTStop, RTPWrkTInit

class TTSSMarkerSentDoneCB(TTSSMarkerNewSent):
    def __init__(self, done_cb:ray.ObjectRef, tts_sess: 'TTSSession', sync:bool=False):
        super().__init__()
        self.done_cb = done_cb
        self.sync = sync
        self.tts_sess_id = tts_sess.id

    def on_proc(self, tro_self):
        print(f'{monotonic():4.3f}: TTSSMarkerSentDoneCB.on_proc')
        x = self.done_cb()
        if self.sync: ray.get(x)

class TTSRequest():
    text: str
    speaker = None
    done_cb: Optional[ray.ObjectRef]
    def __init__(self, text:str, speaker, done_cb:Optional[ray.ObjectRef]=None):
        self.text = text
        self.speaker = speaker
        self.done_cb = done_cb

class TTSSession(InfernWrkThread):
    debug = True
    id: UUID
    tts = None
    worker: RemoteRTPGenFromId
    next_sentence_q: Queue[TTSRequest]

    def __init__(self, tts):
        super().__init__()
        self.id = uuid4()
        self.tts = tts

    def start(self, rtp_actr, rtp_sess_id):
        worker = RemoteRTPGenFromId(rtp_actr, rtp_sess_id)
        self.state_lock.acquire()
        assert self.get_state(locked=True) == RTPWrkTInit
        self.worker = worker
        self.next_sentence_q = Queue()
        self.state_lock.release()
        super().start()

    def run(self):
        super().thread_started()
        while self.get_state() != RTPWrkTStop:
            sent = self.next_sentence_q.get()
            if sent is None: break
            sents = sent.text.split('|')
            for i, p in enumerate(sents):
                if self.get_state() == RTPWrkTStop: break
                print(f'{monotonic():4.3f}: Playing', p)
                self.tts.tts_rt(p, self.worker.soundout, sent.speaker)
                if i < len(sents) - 1:
                    self.worker.soundout(TTSSMarkerNewSent())
            if sent.done_cb is not None:
                self.worker.soundout(TTSSMarkerSentDoneCB(sent.done_cb, self, sync=True))
        if self.get_state() == RTPWrkTStop:
            self.worker.end()

        self.worker.soundout(TTSSMarkerEnd())
        self.worker.join()
        del self.worker

    def say(self, text, done_cb:Optional[ray.ObjectRef]):
        print(f'{monotonic():4.3f}: TTSSession.say')
        speaker = self.tts.get_rand_voice()
        self.next_sentence_q.put(TTSRequest(text, speaker, done_cb=done_cb))

    def stop(self):
        self.next_sentence_q.put(None)
        super().stop()

    def __del__(self):
        if self.debug:
            print('TTSSession.__del__')
