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

from time import monotonic
from uuid import uuid4, UUID
from queue import Queue

from TTSRTPOutput import TTSSMarkerEnd, TTSSMarkerNewSent
from Cluster.RemoteRTPGen import RemoteRTPGen
from Core.InfernWrkThread import InfernWrkThread, RTPWrkTStop

class TTSSMarkerNewSentCB(TTSSMarkerNewSent):
    def __init__(self, tts_actr, tts_sess):
        super().__init__()
        self.tts_actr = tts_actr
        self.tts_sess_id = tts_sess.id

    def on_proc(self, tro_self):
        print(f'{monotonic():4.3f}: TTSSMarkerNewSentCB.on_proc')
        self.tts_actr.tts_session_next_sentence.remote(self.tts_sess_id)

class TTSSession(InfernWrkThread):
    debug = True
    id: UUID
    tts = None
    ptime = 0.030
    worker: RemoteRTPGen
    eos_m: TTSSMarkerNewSentCB
    next_sentence_q: Queue

    def __init__(self, tts, sess_term):
        super().__init__()
        self.id = uuid4()
        self.tts = tts
        self.sess_term = sess_term

    def start(self, tts_actr, rtp_actr, text, target):
        self.state_lock.acquire()
        self.worker = RemoteRTPGen(rtp_actr, target)
        self.text = text
        self.eos_m = TTSSMarkerNewSentCB(tts_actr, self)
        self.next_sentence_q = Queue()
        self.state_lock.release()
        super().start()
        return self.worker.rtp_address

    def run(self):
        super().thread_started()
        if type(self.text) == str:
            text = (self.text,)
        else:
            text = self.text
        from time import sleep
        disconnected = False
        for i, p in enumerate(text):
            speaker = self.tts.get_rand_voice()
            sents = p.split('|')
            for si, p in enumerate(sents):
                if self.get_state() == RTPWrkTStop:
                    break
                #if si > 0:
                #    from time import sleep
                #    sleep(0.5)
                #    #print('sleept')
                print(f'{monotonic():4.3f}: Playing', p)
                self.tts.tts_rt(p, self.worker.soundout, speaker)
                self.worker.soundout(self.eos_m)
                disconnected = not self.next_sentence_q.get()
                if disconnected: break
                print(f'{monotonic():4.3f}: Done playing', p)
                ##while self.get_state() != RTPWrkTStop and \
                ##        self.worker.data_queue.qsize() > 3:
                ##    sleep(0.3)
                ##print(f'get_frm_ctrs={self.worker.get_frm_ctrs()}')
                ##print(f'data_queue.qsize()={self.worker.data_queue.qsize()}')
            if disconnected: break
        ##while True:
        ##    if self.get_state() == RTPWrkTStop:
        ##        self.worker.end()
        ##        break
        ##    if self.worker.data_queue.qsize() > 0:
        ##        sleep(0.1)
        ##        continue
        ##    cntrs = self.worker.get_frm_ctrs()
        ##    if cntrs[0] < cntrs[1]:
        ##        print(f'{cntrs[0]} < {cntrs[1]}')
        ##        sleep(0.01)
        ##        continue
        ##    break
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
        print(f'{monotonic():4.3f}: TTSSession.next_sentence')
        self.next_sentence_q.put(True)

    def stop(self):
        self.next_sentence_q.put(False)
        super().stop()

    def __del__(self):
        if self.debug:
            print('TTSSession.__del__')
