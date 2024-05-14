# Copyright (c) 2018-2024 Sippy Software, Inc. All rights reserved.
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

from typing import Optional, Dict
from weakref import WeakValueDictionary
from queue import Queue
from threading import Lock

from sippy.SipConf import SipConf
from sippy.SipTransactionManager import SipTransactionManager
from sippy.SipURL import SipURL
from sippy.SipRegistrationAgent import SipRegistrationAgent
from sippy.misc import local4remote

#from Core.InfernConfig import InfernConfig

from .InfernUAS import InfernLazyUAS
from .InfernUAC import InfernUAC
from .InfernUA import InfernUA, InfernSIPConf
from .InfernSIPProfile import InfernSIPProfile
from .RemoteSession import RemoteSessionOffer, NewRemoteSessionRequest

from utils.tts import human_readable_time, hal_set, smith_set, \
        bender_set

def good(*a):
    #ED2.breakLoop(0)
    pass

def bad(*a):
    #ED2.breakLoop(1)
    pass

class InfernSIP():
    _c: Dict[str, InfernSIPProfile]
    ua = None
    body = None
    ragent = None
    sip_actr = None
    sippy_c = None
    sessions: WeakValueDictionary
    sessions_lock: Lock

    def __init__(self, sip_actr:'InfernSIPActor', rtp_actr, inf_c:'InfernConfig'):
        sip_c = inf_c.sip_conf
        self.sippy_c = {'_sip_address':sip_c.laddr,
                        '_sip_port':sip_c.lport,
                        '_sip_logger':sip_c.logger}
        self.sip_actr, self.rtp_actr = sip_actr, rtp_actr
        self.sessions = WeakValueDictionary()
        self.session_lock = Lock()
        udsc, udsoc = SipTransactionManager.model_udp_server
        udsoc.nworkers = 1
        SipConf.my_uaname = 'Infernos'
        stm =  SipTransactionManager(self.sippy_c, self.recvRequest)
        self.sippy_c['_sip_tm'] = stm
        #raise Exception(f'{inf_c.connectors}')
        self._c = inf_c.connectors
        for n, v in self._c.items():
            if not v.register: continue
            proxy, port = v.nh_addr
            aor = SipURL(username = v.cli, host = proxy, port = port)
            caddr = local4remote(proxy)
            cport = self.sippy_c['_sip_port']
            contact = SipURL(username = v.cli, host = caddr, port = cport)
            ragent = SipRegistrationAgent(self.sippy_c, aor, contact,
                    user=v.authname, passw=v.authpass,
                    rok_cb=good, rfail_cb=bad)
            ragent.rmsg.getHFBody('to').getUrl().username = v.cli
            ragent.doregister()

    def recvRequest(self, req, sip_t):
        if req.getMethod() in ('NOTIFY', 'PING'):
            # Whynot?
            return (req.genResponse(200, 'OK'), None, None)
        if req.getMethod() == 'INVITE':
            #if self.rserv != None:
            #    return (req.genResponse(486, 'Busy Here'), None, None)
            # New dialog
            source = req.getSource()
            for n, sip_prof in self._c.items():
                assert type(source) == type(sip_prof.nh_addr)
                if source == sip_prof.nh_addr:
                    break
            else:
                return (req.genResponse(500, 'Nobody is home'), None, None)
            isess = InfernLazyUAS(self, sip_prof, req, sip_t)
            with self.session_lock:
                self.sessions[isess.id] = isess
            rso = RemoteSessionOffer(self, isess)
            sip_prof.new_sess_offer(rso)
            return
        return (req.genResponse(501, 'Not Implemented'), None, None)

    def new_session(self, msg:NewRemoteSessionRequest, rval:Optional[Queue]=None):
        uac = InfernUAC(self, msg)
        with self.session_lock:
            self.sessions[uac.id] = uac
        ret = (uac, uac.rsess)
        if rval is None: return ret
        rval.put(ret)

    def get_session(self, sip_sess_id) -> InfernUA:
        with self.session_lock:
            return self.sessions[sip_sess_id]

#    def getPrompts(self):
#        return [f'{human_readable_time()}',] + list(self.prompts)
