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

from threading import Thread, Lock

RTPWrkTInit = 0
RTPWrkTRun = 1
RTPWrkTStop = 2

class InfernWrkThread(Thread):
    state_lock: Lock = None
    state: int = RTPWrkTInit

    def __init__(self):
        self.state_lock = Lock()
        super().__init__()
        self.setDaemon(True)

    def start(self):
        super().start()

    def get_state(self, locked=False):
        if not locked: self.state_lock.acquire()
        state = self.state
        if not locked: self.state_lock.release()
        return state

    def _set_state(self, newstate, expected_state = None, raise_on_error = True):
        self.state_lock.acquire()
        pstate = self.state
        if expected_state is not None and self.state != expected_state:
            self.state_lock.release()
            if raise_on_error:
                raise AssertionError(f'Unexpected state: {self.state}, {expected_state} expected')
            return pstate
        self.state = newstate
        self.state_lock.release()
        return pstate

    def thread_started(self):
        self._set_state(RTPWrkTRun, expected_state = RTPWrkTInit)

    def stop(self):
        pstate = self._set_state(RTPWrkTStop, expected_state = RTPWrkTRun, raise_on_error = True)
        if pstate == RTPWrkTRun:
            self.join()
        self._set_state(RTPWrkTInit, expected_state = RTPWrkTStop)
