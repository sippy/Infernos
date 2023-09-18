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
from sippy.CCEvents import CCEventDisconnect, CCEventTry
from sippy.CCEvents import CCEventRing, CCEventConnect, CCEventPreConnect
from sippy.SipTransactionManager import SipTransactionManager
from sippy.SipCiscoGUID import SipCiscoGUID
from sippy.SipCallId import SipCallId
from sippy.MsgBody import MsgBody
from sippy.SdpOrigin import SdpOrigin
from sippy.Udp_server import Udp_server, Udp_server_opts
from sippy.SipURL import SipURL
from sippy.SipRegistrationAgent import SipRegistrationAgent
from sippy.SipConf import SipConf

from sippy.misc import local4remote

from .InfernRTPGen import InfernRTPGen
import sys

from TTS import TTS
from tts_utils import human_readable_time, hal_set, smith_set, t900_set

body_txt = 'v=0\r\n' + \
  'o=- 380960 380960 IN IP4 192.168.22.95\r\n' + \
  's=-\r\n' + \
  'c=IN IP4 192.168.22.95\r\n' + \
  't=0 0\r\n' + \
  'm=audio 16474 RTP/AVP 0\r\n' + \
  'a=rtpmap:0 PCMU/8000\r\n' + \
  'a=ptime:30\r\n' + \
  'a=sendrecv\r\n' + \
  '\r\n'

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

prompts = [f'Welcome to Infernos.|{human_readable_time()}',] + smith_set() + hal_set() #+ t900_set() 

class InfernUAS(object):
    _o = None
    ua = None
    body = None
    rgen = None
    rserv = None
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
        stm =  SipTransactionManager(self.sippy_c, self.recvRequest)
        self.sippy_c['_sip_tm'] = stm
        self.body = MsgBody(body_txt)
        self.body.parse()
        proxy, port = self.sippy_c['nh_addr']
        aor = SipURL(username = iao.cli, host = proxy, port = port)
        caddr = local4remote(proxy)
        cport = self.sippy_c['_sip_port']
        contact = SipURL(username = iao.cli, host = caddr, port = cport)
        ragent = SipRegistrationAgent(self.sippy_c, aor, contact, user = iao.authname, passw = iao.authpass)
        ragent.rmsg.getHFBody('to').getUrl().username = iao.cld
        ragent.doregister()

    def sess_term(self, ua=None, rtime=None, origin=None, result=0):
        print('disconnected')
        if self.rgen is None:
            return
        self.rgen.stop()
        self.rserv.shutdown()
        self.rserv = None
        self.rgen = None
        if ua is None:
            self.uaA.disconnect()

    def rtp_received(self, data, address, udp_server, rtime):
        pass

    def recvRequest(self, req, sip_t):
        if req.getMethod() in ('NOTIFY', 'PING'):
            # Whynot?
            return (req.genResponse(200, 'OK'), None, None)
        if req.getMethod() == 'INVITE':
            if self.rserv != None:
                return (req.genResponse(486, 'Busy Here'), None, None)
            # New dialog
            self.uaA = UA(self.sippy_c, self.recvEvent, disc_cbs = (self.sess_term,))
            self.uaA.recvRequest(req, sip_t)
            return
        return (req.genResponse(501, 'Not Implemented'), None, None)

    def recvEvent(self, event, ua):
        if isinstance(event, CCEventTry):
            cId, cli, cld, sdp_body, auth, caller_name = event.getData()
            if sdp_body == None:
                return
            sdp_body.parse()
            sect = sdp_body.content.sections[0]
            rtp_target = (sect.c_header.addr, sect.m_header.port)
            rtp_laddr = local4remote(rtp_target[0])
            rserv_opts = Udp_server_opts((rtp_laddr, 0), self.rtp_received)
            rserv_opts.nworkers = 1
            self.rserv = Udp_server({}, rserv_opts)
            self.rgen = InfernRTPGen(self.tts, self.sess_term)
            self.rgen.dl_file = 'Infernos.check.wav'
            self.rgen.start(prompts, self.rserv, rtp_target)
            sect = self.body.content.sections[0]
            sect.c_header.addr = self.rserv.uopts.laddress[0]
            sect.m_header.port = self.rserv.uopts.laddress[1]
            self.body.content.o_header = SdpOrigin()
            oevent = CCEventConnect((200, 'OK', self.body))
            ua.recvEvent(oevent)
            return
