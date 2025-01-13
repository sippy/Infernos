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

from typing import Optional, Union, List, Tuple, Dict
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
    text: Union[str,List[str],Tuple[str]]
    speaker_id: Optional[int]
    done_cb: Optional[callable]
    def __init__(self, text:Union[str,List[str],Tuple[str]], speaker_id:Optional[int]=None, done_cb:Optional[callable]=None):
        self.text = text
        self.speaker_id = speaker_id
        self.done_cb = done_cb

class TTSSndDispatch():
    id: UUID
    debug: bool = False
    cancelled: bool = False
    done_cb: Optional[callable] = None
    cleanup_cb: Optional[callable] = None
    soundout: callable
    output_sr: int
    def __init__(self, soundout:callable, output_sr:int, done_cb:Optional[callable]):
        self.id = uuid4()
        self.soundout, self.output_sr, self.done_cb = soundout, output_sr, done_cb

    def cancel(self):
        self.cancelled = True
        chunk = ASMarkerNewSent() if self.done_cb is None \
                                  else ASMarkerSentDoneCB(self.done_cb, sync=True)
        self.soundout(chunk=chunk)
        if self.cleanup_cb is not None:
            self.cleanup_cb()

    def sound_dispatch(self, chunk):
        if self.cancelled:
            return
        do_cleanup = False
        if chunk is None:
            if self.debug:
                print(f'{monotonic():4.3f}: TTSSndDispatch.sound_dispatch {self.done_cb=}')
            chunk = ASMarkerNewSent() if self.done_cb is None \
                                      else ASMarkerSentDoneCB(self.done_cb, sync=True)
            do_cleanup = True
        elif not isinstance(chunk, ASMarkerGeneric):
            assert chunk.size(0) > 0
            chunk=AudioChunk(chunk, self.output_sr)
        self.soundout(chunk=chunk)
        if do_cleanup and self.cleanup_cb is not None:
            self.cleanup_cb()

class TTSSession():
    debug = True
    id: UUID
    tts: InfernTTSWorker
    tts_actr: ray.remote
    soundout: callable
    active_req: Dict[UUID, TTSSndDispatch]

    def __init__(self, tts:InfernTTSWorker, tts_actr:ray.remote):
        super().__init__()
        self.id = uuid4()
        self.tts, self.tts_actr = tts, tts_actr
        self.active_req = {}

    def start(self, soundout:callable):
        self.soundout = soundout

    def say(self, req:TTSRequest) -> UUID:
        if self.debug:
            print(f'{monotonic():4.3f}: TTSSession.say: ${req.text=}, {req.speaker_id=}, {req.done_cb=}')
        if req.speaker_id is not None:
            speaker = self.tts.get_voice(req.speaker_id)
        else:
            speaker, req.speaker_id = self.tts.get_rand_voice()
        if isinstance(req.text, str): req.text = (req.text,)
        text, done_cb = req.text[0], req.done_cb
        if len(req.text) > 1:
            req.text.pop(0)
            done_cb = partial(self.tts_actr.tts_session_say.remote, rgen_id=self.id, req=req)
        trd = TTSSndDispatch(self.soundout, self.tts.output_sr, done_cb)
        def cleanup_cb():
            if self.debug:
                print(f'{monotonic():4.3f}: TTSSession.cleanup_cb')
            del self.active_req[trd.id]
        trd.cleanup_cb = cleanup_cb
        preq = HelloSippyPlayRequest(self.id, text, speaker, trd.sound_dispatch)
        self.active_req[trd.id] = trd
        self.tts.infer(preq)
        return trd.id

    def stop_saying(self, rsay_id:UUID):
        if self.debug:
            print(f'{monotonic():4.3f}: TTSSession.stop_saying: {rsay_id=}')
        trd = self.active_req.get(rsay_id)
        if trd is None:
            return False
        trd.cancel()
        return True

    def stop(self):
        pass

    def __del__(self):
        if self.debug:
            print('TTSSession.__del__')
