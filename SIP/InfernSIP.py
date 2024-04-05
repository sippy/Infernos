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

from typing import Optional
from weakref import WeakValueDictionary
from queue import Queue

from sippy.SipConf import SipConf
from sippy.SipTransactionManager import SipTransactionManager
from sippy.SipURL import SipURL
from sippy.SipRegistrationAgent import SipRegistrationAgent
from sippy.misc import local4remote

from config.InfernGlobals import InfernGlobals as IG
from .InfernUAS import InfernTTSUAS
from .InfernUAC import InfernUAC

from Core.T2T.Translator import Translator

from utils.tts import human_readable_time, hal_set, smith_set, \
        bender_set

def good(*a):
    #ED2.breakLoop(0)
    pass

def bad(*a):
    #ED2.breakLoop(1)
    pass

class InfernSIP(object):
    _o = None
    ua = None
    body = None
    ragent = None
    sip_actr = None
    tts_actr = None
    stt_actr = None
    sippy_c = None
    sessions: WeakValueDictionary
    tts_lang: str
    stt_lang: str

    def __init__(self, sip_actr, tts_actr, stt_actr, rtp_actr, iao, tts_lang, stt_lang):
        self.sippy_c = {'nh_addr':tuple(iao.nh_addr),
                        '_sip_address':iao.laddr,
                        '_sip_port':iao.lport,
                        '_sip_logger':iao.logger}
        self.sip_actr, self.tts_actr, self.stt_actr, self.rtp_actr = sip_actr, tts_actr, stt_actr, rtp_actr
        self.sessions = WeakValueDictionary()
        self._o = iao
        udsc, udsoc = SipTransactionManager.model_udp_server
        udsoc.nworkers = 1
        SipConf.my_uaname = 'Infernos'
        stm =  SipTransactionManager(self.sippy_c, self.recvRequest)
        self.sippy_c['_sip_tm'] = stm
        proxy, port = self.sippy_c['nh_addr']
        aor = SipURL(username = iao.cli, host = proxy, port = port)
        caddr = local4remote(proxy)
        cport = self.sippy_c['_sip_port']
        contact = SipURL(username = iao.cli, host = caddr, port = cport)
        ragent = SipRegistrationAgent(self.sippy_c, aor, contact,
                user=iao.authname, passw=iao.authpass,
                rok_cb=good, rfail_cb=bad)
        ragent.rmsg.getHFBody('to').getUrl().username = iao.cld
        prompts = ['Welcome to Infernos.'] + bender_set(2) + \
                   smith_set() + hal_set() #+ t900_set()
        if tts_lang[0] != 'en':
            tr = IG.get_translator('en', tts_lang[0])
            prompts = [tr.translate(p) for p in prompts]
        self.tts_lang, self.stt_lang = tts_lang, stt_lang
        self.prompts = prompts
        ragent.doregister()

    def recvRequest(self, req, sip_t):
        if req.getMethod() in ('NOTIFY', 'PING'):
            # Whynot?
            return (req.genResponse(200, 'OK'), None, None)
        if req.getMethod() == 'INVITE':
            #if self.rserv != None:
            #    return (req.genResponse(486, 'Busy Here'), None, None)
            # New dialog
            isess = InfernTTSUAS(self, req, sip_t)
            self.sessions[isess.id] = isess
            return
        return (req.genResponse(501, 'Not Implemented'), None, None)

    def new_session(self, cld:str, rval:Optional[Queue]=None):
        uac = InfernUAC(self, cld)
        self.sessions[uac.id] = uac
        ret = (uac, uac.rsess.get_soundout())
        if rval is None: return ret
        rval.put(ret)

    def get_session(self, sip_sess_id):
        return self.sessions[sip_sess_id]

    def getPrompts(self):
        return [f'{human_readable_time()}',] + self.prompts
