"""Unknown codecs through the real sippy UA states; no Ray, RTP or network."""

import importlib
import pickle
from queue import Queue
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from ray.exceptions import RayTaskError
from sippy.CCEvents import CCEventFail, CCEventUpdate
from sippy.MsgBody import MsgBody
from sippy.SipRequest import SipRequest
from sippy.UasStateIdle import UasStateIdle

from Cluster.RemoteRTPGen import RTPGenError
from Core.Exceptions.InfernSessRejectedErr import InfernSessRejectedErr
from SIP.InfernUA import model_body, uas_failure
from SIP.InfernUAC import InfernUAC
from SIP.InfernUAS import InfernLazyUAS, InfernUAS
from SIP.RemoteSession import RemoteSessionAccept

uas_module = importlib.import_module('SIP.InfernUAS')
sip_actor_module = importlib.import_module('Cluster.InfernSIPActor')
echo_module = importlib.import_module('Apps.Echo.EchoActor')
SIPActor = sip_actor_module.InfernSIPActor.__ray_metadata__.modified_class
EchoActor = echo_module.EchoActor.__ray_metadata__.modified_class

G7221_SDP = '\r\n'.join((
    'v=0',
    'o=- 4002689806 4002689806 IN IP4 192.168.23.235',
    's=pjmedia',
    'b=AS:67',
    't=0 0',
    'a=X-nat:0',
    'm=audio 4000 RTP/AVP 96 97 98 101',
    'c=IN IP4 192.168.23.235',
    'b=TIAS:48000',
    'a=rtcp:4001 IN IP4 192.168.23.235',
    'a=sendrecv',
    'a=rtpmap:96 G7221/32000',
    'a=fmtp:96 bitrate=24000',
    'a=rtpmap:97 G7221/32000',
    'a=fmtp:97 bitrate=32000',
    'a=rtpmap:98 G7221/32000',
    'a=fmtp:98 bitrate=48000',
    'a=rtpmap:101 telephone-event/32000',
    'a=fmtp:101 0-16',
    'a=ssrc:1230052143 cname:46b2749238b427ba',
    '')) + '\r\n'


def invite(sdp):
    return SipRequest('\r\n'.join((
        'INVITE sip:123456@192.168.23.109 SIP/2.0',
        'Via: SIP/2.0/UDP 192.168.23.235:6667;rport;branch=z9hG4bKPj29e5e6d3',
        'Max-Forwards: 70',
        'From: "Fuco Braga" <sip:16043232234@192.168.23.109>;tag=477230c0',
        'To: <sip:123456@192.168.23.109>',
        'Contact: "Fuco Braga" <sip:16043232234@192.168.23.235:6667;ob>',
        'Call-ID: fa343afe8efc4f1ba26a44799a253b2b',
        'CSeq: 28075 INVITE',
        'Content-Type: application/sdp',
        f'Content-Length: {len(sdp)}',
        '', sdp)))


class IncomingCallTests(unittest.TestCase):
    def setUp(self):
        self.tm = Mock()
        self.sip_stack = SimpleNamespace(
            sip_actr=Mock(), rtp_actr=Mock(),
            sippy_c={'_sip_tm': self.tm, '_sip_address': '127.0.0.1', '_sip_port': 5060})
        self.create_rtp = self.enterContext(patch.object(uas_module, 'RemoteRTPGen'))

    def accept(self, sdp):
        sip_t = SimpleNamespace(noack_cb=None, cancel_cb=None, compact=False)
        ua = InfernLazyUAS(self.sip_stack, None, invite(sdp), sip_t)
        self.addCleanup(self.kill_timers, ua)
        rval = Queue()
        ua.accept(RemoteSessionAccept(disc_cb=Mock(), auto_answer=True), rval)
        return ua, rval.get_nowait()

    @staticmethod
    def kill_timers(ua):
        ua.event_cb = None

    def responses(self):
        return [call.args[0] for call in self.tm.sendResponse.call_args_list]

    def test_g7221_only_offer_gets_a_single_488(self):
        ua, rsess = self.accept(G7221_SDP)
        self.assertIsNone(rsess)
        self.create_rtp.assert_not_called()
        final = [r for r in self.responses() if r.getSCode()[0] >= 200]
        self.assertEqual([r.getSCode()[:2] for r in final], [(488, 'Not Acceptable Here')])
        self.assertEqual(ua.state.sname, 'Failed')
        self.assertEqual(str(final[0].getHFBody('reason')),
                         'SIP; cause=488; text="Unsupported audio codec, sorry"')

    def test_no_sdp_gets_a_single_488(self):
        ua, rsess = self.accept('')
        self.assertIsNone(rsess)
        final = [r for r in self.responses() if r.getSCode()[0] >= 200]
        self.assertEqual([r.getSCode()[0] for r in final], [488])

    def test_known_codec_next_to_unknown_ones_is_answered(self):
        self.create_rtp.return_value = SimpleNamespace(
            rtp_address=('127.0.0.1', 24000), sess_id=uuid4())
        sdp = G7221_SDP.replace('RTP/AVP 96 97 98 101', 'RTP/AVP 96 8 101')
        ua, rsess = self.accept(sdp)
        self.assertIs(rsess, self.create_rtp.return_value)
        final = self.responses()[-1]
        self.assertEqual(final.getSCode()[0], 200)
        self.assertEqual(final.getBody().content.sections[0].m_header.formats, [8])
        self.assertEqual(ua.state.sname, 'Connected')
        ua.state = None  # Skip keep-alive and BYE on garbage collection.

    def test_reinvite_to_unknown_codec_keeps_call_and_media(self):
        self.create_rtp.return_value = SimpleNamespace(
            rtp_address=('127.0.0.1', 24000), sess_id=uuid4(), update=Mock())
        sdp = G7221_SDP.replace('RTP/AVP 96 97 98 101', 'RTP/AVP 0')
        ua, rsess = self.accept(sdp)
        ua.changeState((ua.UasStateUpdating,))
        ua.sendUasResponse = Mock()
        ua.outEvent(CCEventUpdate(MsgBody(G7221_SDP)), ua)
        self.assertEqual(ua.sendUasResponse.call_args.args[:2], (488, 'Not Acceptable Here'))
        rsess.update.assert_not_called()
        self.assertEqual(ua.state.sname, 'Connected')
        ua.state = None


class AcceptAlwaysDeliversTests(unittest.TestCase):
    """accept() runs on the ED2 thread; new_sess_accept() blocks on its result."""

    def setUp(self):
        self.tm = Mock()
        self.sip_stack = SimpleNamespace(
            sip_actr=Mock(), rtp_actr=Mock(),
            sippy_c={'_sip_tm': self.tm, '_sip_address': '127.0.0.1', '_sip_port': 5060})
        self.create_rtp = self.enterContext(patch.object(uas_module, 'RemoteRTPGen'))
        self.sdp = G7221_SDP.replace('RTP/AVP 96 97 98 101', 'RTP/AVP 0')

    def lazy_ua(self):
        sip_t = SimpleNamespace(noack_cb=None, cancel_cb=None, compact=False)
        ua = InfernLazyUAS(self.sip_stack, None, invite(self.sdp), sip_t)
        self.addCleanup(setattr, ua, 'event_cb', None)
        return ua

    def accept(self, ua):
        rval = Queue()
        ua.accept(RemoteSessionAccept(disc_cb=Mock(), auto_answer=True), rval)
        return rval.get_nowait()

    def finals(self):
        return [c.args[0].getSCode() for c in self.tm.sendResponse.call_args_list
                if c.args[0].getSCode()[0] >= 200]

    def test_rtp_failure_rejects_new_sess_accept_with_a_single_500(self):
        self.create_rtp.side_effect = RTPGenError('new_rtp_session() failed')
        ua = self.lazy_ua()
        actor = SIPActor()
        actor.sip_stack = SimpleNamespace(get_session=lambda sid: ua)
        ed2 = importlib.import_module('sippy.Core.EventDispatcher').ED2
        outcome = []

        def run():
            try:
                outcome.append(actor.new_sess_accept(ua.id, RemoteSessionAccept(auto_answer=True)))
            except Exception as e:
                outcome.append(e)

        # dispatchThreadCallback is what ED2 really runs: it swallows exceptions.
        with patch.object(ed2, 'callFromThread',
                          side_effect=lambda f, *a: ed2.dispatchThreadCallback(f, a)):
            worker = Thread(target=run, daemon=True)
            worker.start()
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), 'new_sess_accept() is stuck on rval.get()')
        self.assertIsInstance(outcome[0], InfernSessRejectedErr)
        self.assertEqual(self.finals(), [(500, 'Server Internal Error')])
        self.assertEqual(ua.state.sname, 'Failed')
        self.assertEqual(str(self.tm.sendResponse.call_args.args[0].getHFBody('reason')),
                         'SIP; cause=500; text="new_rtp_session() failed"')

    def test_failed_accept_keeps_session_id_and_drops_transaction_refs(self):
        self.create_rtp.side_effect = RTPGenError('new_rtp_session() failed')
        ua = self.lazy_ua()
        sess_id = ua.id
        self.assertIsNone(self.accept(ua))
        self.assertEqual(ua.id, sess_id)
        for attr in ('_sip_stack', '_req', '_sip_t', '_id'):
            self.assertFalse(hasattr(ua, attr), attr)

    def test_unexpected_rtp_error_gets_a_single_500(self):
        self.create_rtp.side_effect = RuntimeError('RTP actor is gone')
        ua = self.lazy_ua()
        self.assertIsNone(self.accept(ua))
        self.assertEqual(self.finals(), [(500, 'Server Internal Error')])
        self.assertEqual(ua.state.sname, 'Failed')

    def test_failure_after_rtp_setup_releases_media_and_gets_a_single_500(self):
        rsess = self.create_rtp.return_value = Mock(
            rtp_address=('127.0.0.1', 24000), sess_id=uuid4())
        ua = self.lazy_ua()
        with patch.object(InfernLazyUAS, 'send_uas_resp', side_effect=RuntimeError('boom')):
            self.assertIsNone(self.accept(ua))
        self.assertEqual(self.finals(), [(500, 'Server Internal Error')])
        rsess.end.assert_called_once()
        self.assertIsNone(ua.rsess)

    def test_failure_before_ua_state_setup_still_answers_the_invite(self):
        ua = self.lazy_ua()
        with patch.object(UasStateIdle, 'recvRequest', side_effect=RuntimeError('boom')):
            self.assertIsNone(self.accept(ua))
        self.assertEqual([code for code, _ in self.finals()], [500])

    def test_accept_after_cancel_sends_nothing_but_still_delivers(self):
        ua = self.lazy_ua()
        ua.cancelled()
        self.assertIsNone(self.accept(ua))
        self.assertEqual(self.finals(), [])
        self.create_rtp.assert_not_called()


class FailureEventTests(unittest.TestCase):
    def test_failure_is_handled_by_exact_class_lookup_and_copies(self):
        event = uas_failure('why', code=500)
        self.assertIs(type(event), CCEventFail)
        self.assertEqual(event.getCopy().getData(), (500, 'Server Internal Error'))
        self.assertEqual(str(event.getCopy().reason_rfc3326), 'SIP; cause=500; text="why"')

    def test_unusable_answer_stops_uac_without_updating_media(self):
        from sippy.CCEvents import CCEventConnect
        ua = InfernUAC.__new__(InfernUAC)
        ua.rsess, ua.uas = Mock(), Mock()
        with patch.object(InfernUAC, 'recvEvent') as recv:
            ua.outEvent(CCEventConnect((200, 'OK', MsgBody(G7221_SDP))), ua)
        self.assertEqual(recv.call_args.args[0].getData()[0], 488)
        ua.rsess.update.assert_not_called()
        ua.uas.recvEvent.assert_not_called()


class ActorTests(unittest.TestCase):
    def test_sip_actor_raises_rejection_when_negotiation_fails(self):
        actor = SIPActor()
        ua = SimpleNamespace(accept=lambda msg, rval: rval.put(None))
        actor.sip_stack = SimpleNamespace(get_session=lambda sid: ua)
        ed2 = importlib.import_module('sippy.Core.EventDispatcher').ED2
        with patch.object(ed2, 'callFromThread', side_effect=lambda f, *a: f(*a)):
            with self.assertRaises(InfernSessRejectedErr):
                actor.new_sess_accept(uuid4(), Mock())

    def test_rejection_survives_ray_transport(self):
        cause = pickle.loads(pickle.dumps(InfernSessRejectedErr('no codec')))
        err = RayTaskError('new_sess_accept', 'tb', cause).as_instanceof_cause()
        self.assertIsInstance(err, InfernSessRejectedErr)

    def test_echo_treats_rejection_as_normal_call_outcome(self):
        actor = EchoActor()
        actor.stt_actr, actor.sip_actr = Mock(), Mock()
        actor.echo_actr = Mock()
        offer = SimpleNamespace(sip_sess_id=uuid4(), accept=Mock())
        with patch('Apps.Echo.EchoSession.ray.get',
                   side_effect=InfernSessRejectedErr('no codec')):
            actor.new_sip_session_received(offer)
        self.assertFalse(actor.sessions)
        actor.stt_actr.new_stt_session.remote.assert_not_called()
        actor.sip_actr.sess_term.remote.assert_not_called()


if __name__ == '__main__':
    unittest.main()
