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

from typing import Optional, Union, List, Tuple
from time import monotonic
from uuid import uuid4, UUID
from queue import Queue

import ray

from RTP.RTPOutputWorker import TTSSMarkerEnd, TTSSMarkerNewSent, TTSSMarkerGeneric, \
    TTSSMarkerSentDoneCB
from Core.InfernWrkThread import InfernWrkThread, RTPWrkTStop, RTPWrkTInit

class TTSRequest():
    text: str
    speaker = None
    done_cb: Optional[callable]
    def __init__(self, text:str, speaker, done_cb:Optional[callable]=None):
        self.text = text
        self.speaker = speaker
        self.done_cb = done_cb

class TTSSession(InfernWrkThread):
    debug = True
    id: UUID
    tts = None
    soundout: callable
    next_sentence_q: Queue[TTSRequest]

    def __init__(self, tts):
        super().__init__()
        self.id = uuid4()
        self.tts = tts

    def start(self, soundout:callable):
        self.state_lock.acquire()
        assert self.get_state(locked=True) == RTPWrkTInit
        self.soundout = soundout
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
                self.tts.tts_rt(p, self.soundout_cb, sent.speaker)
                if i < len(sents) - 1:
                    self.soundout(chunk=TTSSMarkerNewSent())
            mark = TTSSMarkerNewSent() if sent.done_cb is None else TTSSMarkerSentDoneCB(sent.done_cb, sync=True)
            self.soundout(chunk=mark)
        self.soundout(chunk=TTSSMarkerEnd())
        del self.soundout

    def soundout_cb(self, chunk):
        if not isinstance(chunk, TTSSMarkerGeneric):
            chunk = chunk.to('cpu')
        res = self.soundout(chunk=chunk)
        if isinstance(res, ray.ObjectRef):
            res = ray.get(res)
        return res

    def say(self, text, speaker, done_cb:Optional[callable]):
        print(f'{monotonic():4.3f}: TTSSession.say')
        speaker = self.tts.get_rand_voice() if speaker is None else self.tts.get_voice(speaker)
        self.next_sentence_q.put(TTSRequest(text, speaker, done_cb=done_cb))

    def stop(self):
        self.next_sentence_q.put(None)
        super().stop()

    def __del__(self):
        if self.debug:
            print('TTSSession.__del__')

from functools import partial
from HelloSippyTTSRT.HelloSippyRTPipe import HelloSippyPlayRequest
from Core.AudioChunk import AudioChunk
from Cluster.InfernTTSWorker import InfernTTSWorker

class TTSSession2():
    debug = True
    id: UUID
    tts: InfernTTSWorker
    tts_actr: ray.remote
    soundout: callable

    def __init__(self, tts:InfernTTSWorker, tts_actr:ray.remote):
        super().__init__()
        self.id = uuid4()
        self.tts, self.tts_actr = tts, tts_actr

    def start(self, soundout:callable):
        self.soundout = soundout

    def sound_dispatch(self, chunk, done_cb:callable):
        if chunk is None:
            print(f'{monotonic():4.3f}: TTSSession.sound_dispatch {done_cb=}')
            chunk = TTSSMarkerNewSent() if done_cb is None else TTSSMarkerSentDoneCB(done_cb, sync=True)
        elif not isinstance(chunk, TTSSMarkerGeneric):
            assert chunk.size(0) > 0
            chunk=AudioChunk(chunk, self.tts.output_sr)
        self.soundout(chunk=chunk)

    def say(self, text:Union[str,List[str],Tuple[str]], speaker_id, done_cb:Optional[callable]):
        if self.debug:
            print(f'{monotonic():4.3f}: TTSSession.say: ${text=}, {speaker_id=}, {done_cb=}')
        if speaker_id is not None:
            speaker = self.tts.get_voice(speaker_id)
        else:
            speaker, speaker_id = self.tts.get_rand_voice()
        if isinstance(text, str): text = (text,)
        if len(text) > 1:
            done_cb = partial(self.tts_actr.tts_session_say.remote, rgen_id=self.id, text=text[1:],
                              speaker_id=speaker_id, done_cb=done_cb)
        soundout = partial(self.sound_dispatch, done_cb=done_cb)
        req = HelloSippyPlayRequest(self.id, text[0], speaker, soundout)
        self.tts.infer(req)

    def stop(self):
        pass

    def __del__(self):
        if self.debug:
            print('TTSSession.__del__')
