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

from sippy.Core.EventDispatcher import ED2
from threading import Thread, Lock

import sys
sys.path.append('..')
from TTS import TTSSMarkerEnd, TTSSMarkerNewSent

RTPGenInit = 0
RTPGenRun = 1
RTPGenSuspend = 2
RTPGenStop = 3

class InfernRTPGen(Thread):
    tts = None
    ptime = 0.030
    state_lock = Lock()
    state = RTPGenInit
    userv = None
    target = None
    worker = None

    def __init__(self, tts, sess_term):
        self.tts = tts
        self.sess_term = sess_term
        super().__init__()
        self.setDaemon(True)

    def start(self, text, userv, target):
        self.state_lock.acquire()
        self.target = target
        self.userv = userv
        if self.state == RTPGenSuspend:
            self.state = RTPGenRun
            self.state_lock.release()
            return
        self.worker = self.tts.start_pkt_proc(self.send_pkt)
        self.text = text
        self.state_lock.release()
        Thread.start(self)

    def send_pkt(self, pkt):
        self.userv.send_to(pkt, self.target)

    def get_state(self):
        self.state_lock.acquire()
        state = self.state
        self.state_lock.release()
        return state

    def run(self):
        self.state_lock.acquire()
        if self.state == RTPGenStop:
            self.state_lock.release()
            return
        self.state = RTPGenRun
        if type(self.text) == str:
            text = (self.text,)
        else:
            text = self.text
        self.state_lock.release()
        for i, p in enumerate(text):
            sents = p.split('|')
            speaker = self.tts.get_rand_voice()
            for si, p in enumerate(sents):
                if self.get_state() == RTPGenStop:
                    break
                if si > 0:
                    from time import sleep
                    sleep(0.5)
                    #print('sleept')
                print('Playing', p)
                self.tts.play_tts(p, self.worker, speaker)
                self.worker.soundout(TTSSMarkerNewSent())
        self.worker.soundout(TTSSMarkerEnd())
        self.worker.join()
        ED2.callFromThread(self.sess_term)
        del self.sess_term
        del self.worker

    def stop(self):
        self.state_lock.acquire()
        pstate = self.state
        if self.state in (RTPGenRun, RTPGenSuspend):
            self.state = RTPGenStop
        self.state_lock.release()
        if pstate in (RTPGenRun, RTPGenSuspend):
            self.join()
        self.userv = None
        self.state_lock.acquire()
        self.state = RTPGenInit
        self.state_lock.release()

    def suspend(self):
        self.state_lock.acquire()
        if self.state == RTPGenRun:
            self.state = RTPGenSuspend
        else:
            etext = 'suspend() is called in the wrong state: %s' % self.state
            self.state_lock.release()
            raise Exception(etext)
        self.state_lock.release()
