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

from sippy.UA import UA
from sippy.CCEvents import CCEventTry, CCEventConnect, CCEventFail
from sippy.SipTransactionManager import SipTransactionManager
from sippy.MsgBody import MsgBody
from sippy.SdpOrigin import SdpOrigin
from sippy.Udp_server import Udp_server, Udp_server_opts
from sippy.SipURL import SipURL
from sippy.SipRegistrationAgent import SipRegistrationAgent
from sippy.SipConf import SipConf
from sippy.SdpMedia import MTAudio
from sippy.SipReason import SipReason

from sippy.misc import local4remote

from .InfernRTPGen import InfernRTPGen

from TTS import TTS
from utils.tts import human_readable_time, hal_set, smith_set, t900_set, \
        bender_set

ULAW_PT = 0
ULAW_RM = 'PCMU/8000'
body_txt = 'v=0\r\n' + \
  'o=- 380960 380960 IN IP4 192.168.22.95\r\n' + \
  's=-\r\n' + \
  'c=IN IP4 192.168.22.95\r\n' + \
  't=0 0\r\n' + \
 f'm=audio 16474 RTP/AVP {ULAW_PT}\r\n' + \
 f'a=rtpmap:0 {ULAW_RM}\r\n' + \
  'a=ptime:30\r\n' + \
  'a=sendrecv\r\n' + \
  '\r\n'
model_body = MsgBody(body_txt)
model_body.parse()

class InfernUASConf(object):
    cli = 'infernos_uas'
    cld = 'infernos_uac'
    authname = None
    authpass = None
    nh_addr = ('192.168.0.102', 5060)
    laddr = None
    lport = None
    logger = None

    def __init__(self):
        self.laddr = SipConf.my_address
        self.lport = SipConf.my_port

prompts = ['Welcome to Infernos.'] + bender_set(2) + \
        smith_set() + hal_set() #+ t900_set() 

from sippy.Core.EventDispatcher import ED2

def good(*a):
    #ED2.breakLoop(0)
    pass

def bad(*a):
    #ED2.breakLoop(1)
    pass

class InfernUASFailure(CCEventFail):
    code = 488
    msg = 'Not Acceptable Here'
    def __init__(self, reason):
        super().__init__((self.code, self.msg))
        self.reason = SipReason(protocol='SIP', cause=self.code,
                                reason=reason)

class InfernTTSUAS(UA):
    _rserv = None
    _rgen = None
    #_rtp_target = None

    def __init__(self, sippy_c, tts, req, sip_t):
        self._rgen = InfernRTPGen(tts, self.sess_term)
        self._rgen.dl_file = 'Infernos.check.wav'
        super().__init__(sippy_c, self.outEvent)
        assert sip_t.noack_cb is None
        sip_t.noack_cb = self.sess_term
        self.recvRequest(req, sip_t)

    def getPrompts(self):
        return [f'{human_readable_time()}',] + prompts

    def outEvent(self, event, ua):
        if not isinstance(event, CCEventTry):
            #ua.disconnect()
            return
        #if isinstance(event, CCEventConnect):
        #    #self._rgen.start(prompts, self._rserv, self._rtp_target)
        #    return
        cId, cli, cld, sdp_body, auth, caller_name = event.getData()
        if sdp_body == None:
            ua.disconnect()
            return
        sdp_body.parse()
        sects = [s for s in sdp_body.content.sections
                 if s.m_header.type == MTAudio and
                 ULAW_PT in s.m_header.formats]
        if len(sects) == 0:
            event = InfernUASFailure("only G.711u audio is supported at this time, sorry")
            self.recvEvent(event)
            return
        sect = sects[0]
        rtp_target = (sect.c_header.addr, sect.m_header.port)
        rtp_laddr = local4remote(rtp_target[0])
        #self._rtp_target = rtp_target
        rserv_opts = Udp_server_opts((rtp_laddr, 0), self.rtp_received)
        rserv_opts.nworkers = 1
        self._rserv = Udp_server({}, rserv_opts)
        #isess = InfernSession(rtp_laddr, self.tts)
        #    isess.uaA = ua
        body = model_body.getCopy()
        sect = body.content.sections[0]
        sect.c_header.addr = self._rserv.uopts.laddress[0]
        sect.m_header.port = self._rserv.uopts.laddress[1]
        body.content.o_header = SdpOrigin()
        oevent = CCEventConnect((200, 'OK', body))
        self.disc_cbs = (self.sess_term,)
        self._rgen.start(self.getPrompts(), self._rserv, rtp_target)
        return self.recvEvent(oevent)

    def rtp_received(self, data, address, udp_server, rtime):
        pass

    def sess_term(self, ua=None, rtime=None, origin=None, result=0):
        print('disconnected')
        if self._rgen is None:
            return
        self._rgen.stop()
        self._rserv.shutdown()
        self._rserv = None
        self._rgen = None
        if ua != self:
            self.disconnect()

class InfernSIP(object):
    _o = None
    ua = None
    body = None
    ragent = None
    tts = None
    sippy_c = None

    def __init__(self, iao):
        self.sippy_c = {'nh_addr':tuple(iao.nh_addr),
                        '_sip_address':iao.laddr,
                        '_sip_port':iao.lport,
                        '_sip_logger':iao.logger}
        self.tts = TTS()
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
        ragent.doregister()

    def recvRequest(self, req, sip_t):
        if req.getMethod() in ('NOTIFY', 'PING'):
            # Whynot?
            return (req.genResponse(200, 'OK'), None, None)
        if req.getMethod() == 'INVITE':
            #if self.rserv != None:
            #    return (req.genResponse(486, 'Busy Here'), None, None)
            # New dialog
            isess = InfernTTSUAS(self.sippy_c, self.tts, req, sip_t)
            return
        return (req.genResponse(501, 'Not Implemented'), None, None)
