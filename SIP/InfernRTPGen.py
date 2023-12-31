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

import sys
from TTS import TTSSMarkerEnd, TTSSMarkerNewSent
from .InfernWrkThread import InfernWrkThread, RTPWrkTInit, RTPWrkTRun, RTPWrkTStop

class InfernRTPGen(InfernWrkThread):
    tts = None
    ptime = 0.030
    userv = None
    target = None
    worker = None
    dl_file = None

    def __init__(self, tts, sess_term):
        super().__init__()
        self.tts = tts
        self.sess_term = sess_term
        self.setDaemon(True)

    def start(self, text, userv, target):
        self.state_lock.acquire()
        self.target = target
        self.userv = userv
        wrkr = self.tts.get_pkt_proc()
        wrkr.set_pkt_send_f(self.send_pkt)
        if self.dl_file is not None:
            wrkr.enable_datalog(self.dl_file)
        wrkr.start()
        self.worker = wrkr
        self.text = text
        self.state_lock.release()
        super().start()

    def send_pkt(self, pkt):
        self.userv.send_to(pkt, self.target)

    def run(self):
        super().thread_started()
        if type(self.text) == str:
            text = (self.text,)
        else:
            text = self.text
        from time import sleep
        for i, p in enumerate(text):
            sents = p.split('|')
            speaker = self.tts.get_rand_voice()
            for si, p in enumerate(sents):
                if self.get_state() == RTPWrkTStop:
                    break
                #if si > 0:
                #    from time import sleep
                #    sleep(0.5)
                #    #print('sleept')
                print('Playing', p)
                self.tts.tts_rt(p, self.worker.soundout,
                                speaker)
                self.worker.soundout(TTSSMarkerNewSent())
                while self.get_state() != RTPWrkTStop and \
                        self.worker.data_queue.qsize() > 3:
                    sleep(0.3)
                print(f'get_frm_ctrs={self.worker.get_frm_ctrs()}')
                print(f'data_queue.qsize()={self.worker.data_queue.qsize()}')
        while True:
            if self.get_state() == RTPWrkTStop:
                self.worker.end()
                break
            if self.worker.data_queue.qsize() > 0:
                sleep(0.1)
                continue
            cntrs = self.worker.get_frm_ctrs()
            if cntrs[0] < cntrs[1]:
                print(f'{cntrs[0]} < {cntrs[1]}')
                sleep(0.01)
                continue
            break

        self.worker.soundout(TTSSMarkerEnd())
        self.worker.join()
        ED2.callFromThread(self.sess_term)
        del self.sess_term
        del self.worker

    def stop(self):
        super().stop()
        self.userv = None
