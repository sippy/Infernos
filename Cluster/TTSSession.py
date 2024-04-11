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

from Core.AStreamMarkers import ASMarkerNewSent, ASMarkerGeneric, \
    ASMarkerSentDoneCB

from functools import partial
from HelloSippyTTSRT.HelloSippyRTPipe import HelloSippyPlayRequest
from Core.AudioChunk import AudioChunk
from Cluster.InfernTTSWorker import InfernTTSWorker

class TTSRequest():
    text: str
    speaker = None
    done_cb: Optional[callable]
    def __init__(self, text:str, speaker, done_cb:Optional[callable]=None):
        self.text = text
        self.speaker = speaker
        self.done_cb = done_cb

class TTSSession():
    debug = False
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
            if self.debug:
                print(f'{monotonic():4.3f}: TTSSession.sound_dispatch {done_cb=}')
            chunk = ASMarkerNewSent() if done_cb is None else ASMarkerSentDoneCB(done_cb, sync=True)
        elif not isinstance(chunk, ASMarkerGeneric):
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
