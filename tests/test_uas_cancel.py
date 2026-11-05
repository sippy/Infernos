"""CANCEL and missing ACK after accept: real InfernSIP and sippy over loopback UDP."""

import importlib
import socket
import time
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from sippy.CCEvents import CCEventConnect, CCEventRing
from sippy.Core.EventDispatcher import ED2
from sippy.MsgBody import MsgBody
from sippy.SipConf import SipConf
from sippy.SipTransactionManager import SipTransactionManager

from SIP.InfernSIP import InfernSIP
from SIP.RemoteSession import RemoteSessionAccept

uas_module = importlib.import_module('SIP.InfernUAS')

SDP = '\r\n'.join(('v=0', 'o=- 1 1 IN IP4 127.0.0.1', 's=-', 'c=IN IP4 127.0.0.1', 't=0 0',
                   'm=audio 4000 RTP/AVP 0', 'a=sendrecv', '')) + '\r\n'


def free_udp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Caller:
    """Just enough of a SIP phone to speak to the transaction manager."""
    def __init__(self, sip_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', 0))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.dst = ('127.0.0.1', sip_port)
        self.call_id = uuid4().hex
        self.branch = 'z9hG4bK' + uuid4().hex
        self.received = []

    def _send(self, method, *extra):
        msg = (f'{method} sip:1@127.0.0.1:{self.dst[1]} SIP/2.0',
               f'Via: SIP/2.0/UDP 127.0.0.1:{self.port};branch={self.branch}',
               'Max-Forwards: 70', 'From: <sip:2@127.0.0.1>;tag=abc123',
               f'To: <sip:1@127.0.0.1:{self.dst[1]}>',
               f'Call-ID: {self.call_id}', f'CSeq: 1 {method}') + extra
        self.sock.sendto('\r\n'.join(msg).encode(), self.dst)

    def invite(self):
        self._send('INVITE', f'Contact: <sip:2@127.0.0.1:{self.port}>',
                   'Content-Type: application/sdp', f'Content-Length: {len(SDP)}', '', SDP)

    def cancel(self):
        self._send('CANCEL', 'Content-Length: 0', '', '')

    def drain(self):
        while True:
            try:
                data = self.sock.recv(65535).decode()
            except BlockingIOError:
                return
            lines = data.split('\r\n')
            cseq = next(x for x in lines if x.lower().startswith('cseq')).split()[-1]
            start = lines[0].split()
            self.received.append((int(start[1]) if start[0] == 'SIP/2.0' else start[0], cseq))

    def saw(self, *messages):
        self.drain()
        return all(m in self.received for m in messages)


class UasLoopbackTests(unittest.TestCase):
    def setUp(self):
        timers = ED2.tlisteners
        ED2.tlisteners = []
        self.addCleanup(setattr, ED2, 'tlisteners', timers)
        self.addCleanup(setattr, SipConf, 'my_uaname', SipConf.my_uaname)
        udsoc = SipTransactionManager.model_udp_server[1]
        self.addCleanup(setattr, udsoc, 'nworkers', udsoc.nworkers)

        sip_port = free_udp_port()
        self.caller = Caller(sip_port)
        self.addCleanup(self.caller.sock.close)
        self.offers = []
        self.disc_cb = Mock()
        self.rsess = SimpleNamespace(rtp_address=('127.0.0.1', 24000), sess_id=uuid4(),
                                     end=Mock(), join=Mock())
        self.enterContext(patch.object(uas_module, 'RemoteRTPGen', return_value=self.rsess))
        prof = SimpleNamespace(nh_addr=('127.0.0.1', self.caller.port), register=False,
                               new_sess_offer=self.offers.append)
        sip_conf = SimpleNamespace(laddr='127.0.0.1', lport=sip_port, logger=Mock())
        self.stack = InfernSIP(Mock(), Mock(),
                               SimpleNamespace(sip_conf=sip_conf, connectors={'sip': prof}))
        self.tm = self.stack.sippy_c['_sip_tm']
        self.addCleanup(self.tm.shutdown)

    def pump(self, predicate, what):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() > deadline:
                self.fail(f'timed out waiting for {what}; caller got {self.caller.received}')
            ED2.loop(0.01)

    def settle(self):
        ED2.loop(0.1)
        self.caller.drain()

    def accept(self, auto_answer):
        self.caller.invite()
        self.pump(lambda: self.offers, 'the session offer')
        ua = self.stack.get_session(self.offers[0].sip_sess_id)
        self.addCleanup(setattr, ua, 'state', None)  # No BYE on garbage collection.
        rval = Queue()
        ua.accept(RemoteSessionAccept(disc_cb=self.disc_cb, auto_answer=auto_answer), rval)
        self.assertIs(rval.get_nowait(), self.rsess)
        return ua

    def assert_session_terminated_once(self, ua):
        self.assertEqual(ua.state.sname, 'Disconnected')
        self.rsess.end.assert_called_once()
        self.rsess.join.assert_called_once()
        self.disc_cb.assert_called_once()
        self.assertIsNone(ua.rsess)

    def cancel_call(self, ua):
        self.caller.cancel()
        self.pump(lambda: self.caller.saw((200, 'CANCEL'), (487, 'INVITE')), 'CANCEL responses')
        self.assert_session_terminated_once(ua)

    def test_cancel_terminates_unanswered_session(self):
        self.cancel_call(self.accept(auto_answer=False))

    def test_cancel_terminates_ringing_session(self):
        ua = self.accept(auto_answer=False)
        ua.recvEvent(CCEventRing((180, 'Ringing', None)))
        self.pump(lambda: self.caller.saw((180, 'INVITE')), 'the 180')
        self.assertEqual(ua.state.sname, 'Ringing(UAS)')
        self.cancel_call(ua)

    def test_answer_arriving_after_cancel_is_ignored(self):
        ua = self.accept(auto_answer=False)
        self.cancel_call(ua)
        ua.recvEvent(CCEventConnect((200, 'OK', MsgBody(SDP))))
        self.settle()
        self.assertEqual(ua.state.sname, 'Disconnected')
        self.assertNotIn((200, 'INVITE'), self.caller.received)

    def test_missing_ack_hangs_up_and_terminates_session_once(self):
        ua = self.accept(auto_answer=True)
        self.pump(lambda: self.caller.saw((200, 'INVITE')), 'the 200')
        transaction = next(t for t in self.tm.tserver.values() if t.method == 'INVITE')
        transaction.teD.cancel()
        self.tm.timerD(transaction)  # Otherwise fires 32 seconds after the 200.
        self.pump(lambda: self.caller.saw(('BYE', 'BYE')), 'the BYE')
        self.assert_session_terminated_once(ua)


if __name__ == '__main__':
    unittest.main()
