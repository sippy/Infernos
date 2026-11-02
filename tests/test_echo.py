"""Offline application contracts; no Ray initialization or model execution."""

import importlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from cerberus import Validator
import torch
from Core.AudioChunk import AudioChunk
from Cluster.STTSession import STTRequest, STTResult, STTSentinel
from Apps.Echo.EchoAppConfig import EchoAppConfig
from Apps.Echo.EchoProfile import EchoProfile

actor_module = importlib.import_module('Apps.Echo.EchoActor')
session_module = importlib.import_module('Apps.Echo.EchoSession')
config_module = importlib.import_module('Core.InfernConfig')
Actor = actor_module.EchoActor.__ray_metadata__.modified_class
ROOT = Path(__file__).resolve().parents[1]


def remote_actor(**methods):
    return SimpleNamespace(**{name: SimpleNamespace(remote=Mock(side_effect=func))
                              for name, func in methods.items()})


class EchoTests(unittest.TestCase):
    def setUp(self):
        # Resolve fake remote results locally; never start a Ray service.
        self.enterContext(patch.object(session_module.ray, 'get', side_effect=lambda value: value))
        self.enterContext(patch.object(session_module.ray, 'init', side_effect=AssertionError('Ray init forbidden')))
        self.actor = Actor()
        self.actor.echo_actr = remote_actor(
            text_in=self.actor.text_in, tts_say_done=self.actor.tts_say_done,
            sess_term=self.actor.sess_term)
        self.stt_id, self.tts_id, self.rtp_id = uuid4(), uuid4(), uuid4()
        self.actor.stt_actr = remote_actor(
            new_stt_session=lambda **kw: self.stt_id,
            stt_session_soundin=lambda **kw: None,
            stt_session_end=lambda sid: None)
        self.actor.tts_actr = remote_actor(
            new_tts_session=lambda: self.tts_id,
            tts_session_start=lambda *args: None,
            tts_session_say=lambda **kw: uuid4(),
            tts_session_end=lambda sid: None)
        self.actor.sip_actr = remote_actor(sess_term=lambda sid: None)
        self.actor.stt_lang = 'de'
        self.actor.translator = None
        self.rtp = remote_actor(rtp_session_connect=lambda *args: None,
                                rtp_session_soundout=lambda *args, **kw: None)
        self.offer = SimpleNamespace(sip_sess_id=uuid4(), accept=Mock(
            return_value=(self.rtp, self.rtp_id)))

    def start_session(self):
        self.actor.new_sip_session_received(self.offer)
        return next(iter(self.actor.sessions.values()))

    def result(self, text='Guten Tag', no_speech_prob=0.1):
        chunk = AudioChunk(torch.zeros(160), 16000)
        return STTResult(text, no_speech_prob, STTRequest(chunk, None, 'de'))

    def test_accept_vad_stt_callback_tts_and_rtp_output(self):
        session = self.start_session()
        msg = self.offer.accept.call_args.kwargs['msg']
        self.assertTrue(msg.auto_answer)
        connect = self.rtp.rtp_session_connect.remote.call_args.args
        self.assertEqual(connect[0], self.rtp_id)
        ain = connect[1]
        self.assertIsNone(ain.audio_in)
        self.assertTrue(callable(ain.vad_chunk_in))
        chunk = AudioChunk(torch.zeros(160), 16000)
        ain.vad_chunk_in(chunk=chunk)
        call = self.actor.stt_actr.stt_session_soundin.remote.call_args
        self.assertEqual(call.kwargs['sess_id'], self.stt_id)
        req = call.kwargs['req']
        self.assertIs(req.chunk, chunk)
        self.assertEqual((req.lang, req.mode), ('de', 'transcribe'))
        req.text_cb(result=self.result())
        say = self.actor.tts_actr.tts_session_say.remote.call_args.kwargs
        self.assertEqual(say['rgen_id'], self.tts_id)
        self.assertEqual(say['req'].text, 'Guten Tag')
        self.assertIsNone(say['req'].speaker_id)
        output = self.actor.tts_actr.tts_session_start.remote.call_args.args[1]
        output(chunk=chunk)
        self.rtp.rtp_session_soundout.remote.assert_called_once_with(self.rtp_id, chunk=chunk)
        self.assertEqual(chunk.track_id, 1)
        say['req'].done_cb()
        self.assertEqual(len(session.say_buffer), 0)
        msg.disc_cb()
        self.assertFalse(self.actor.sessions)
        self.actor.stt_actr.stt_session_end.remote.assert_called_once_with(self.stt_id)
        self.actor.tts_actr.tts_session_end.remote.assert_called_once_with(self.tts_id)
        self.actor.sip_actr.sess_term.remote.assert_not_called()
        # Late queued callbacks and duplicate disconnects are harmless.
        req.text_cb(result=self.result())
        say['req'].done_cb()
        msg.disc_cb()

    def check_sippy_disconnect(self, event_args):
        session = self.start_session()
        msg = self.offer.accept.call_args.kwargs['msg']
        ain = self.rtp.rtp_session_connect.remote.call_args.args[1]
        ain.vad_chunk_in(chunk=AudioChunk(torch.zeros(160), 16000))
        text_cb = self.actor.stt_actr.stt_session_soundin.remote.call_args.kwargs['req'].text_cb
        text_cb(result=self.result())
        done_cb = self.actor.tts_actr.tts_session_say.remote.call_args.kwargs['req'].done_cb
        msg.disc_cb(*event_args)
        self.assertNotIn(session.id, self.actor.sessions)
        self.assertTrue(session.closed)
        self.assertFalse(session.say_buffer)
        self.actor.stt_actr.stt_session_end.remote.assert_called_once_with(self.stt_id)
        self.actor.tts_actr.tts_session_end.remote.assert_called_once_with(self.tts_id)
        self.actor.sip_actr.sess_term.remote.assert_not_called()
        # Repeated SIP events and in-flight STT/playback callbacks are benign.
        msg.disc_cb(*event_args)
        text_cb(result=self.result())
        done_cb()
        self.actor.stt_actr.stt_session_end.remote.assert_called_once()
        self.actor.tts_actr.tts_session_end.remote.assert_called_once()
        self.actor.tts_actr.tts_session_say.remote.assert_called_once()
        self.actor.sip_actr.sess_term.remote.assert_not_called()

    def test_sippy_disconnect_three_positional_arguments(self):
        self.check_sippy_disconnect((object(), 123.0, 'caller'))

    def test_sippy_disconnect_four_positional_arguments(self):
        self.check_sippy_disconnect((object(), 123.0, 'caller', 0))

    def test_disconnect_adapter_is_serializable(self):
        from functools import partial
        from ray import cloudpickle
        callback = partial(session_module._sess_term, sterm=dict,
                           sess_id='echo-id', sip_sess_id='sip-id')
        restored = cloudpickle.loads(cloudpickle.dumps(callback))
        self.assertEqual(restored(object(), 123.0, 'caller', 0),
                         {'sess_id': 'echo-id', 'sip_sess_id': 'sip-id', 'relaxed': True})

    def test_translation_and_serial_playback(self):
        self.actor.translator = Mock(side_effect=lambda text: 'translated ' + text)
        session = self.start_session()
        self.actor.text_in(session.id, self.result('eins'))
        self.actor.text_in(session.id, self.result('zwei'))
        say = self.actor.tts_actr.tts_session_say.remote
        self.assertEqual(say.call_count, 1)
        self.assertEqual(say.call_args.kwargs['req'].text, 'translated eins')
        say.call_args.kwargs['req'].done_cb()
        self.assertEqual(say.call_count, 2)
        self.assertEqual(say.call_args.kwargs['req'].text, 'translated zwei')
        self.assertEqual(self.actor.translator.call_count, 2)

    def test_empty_silence_and_sentinel_do_not_translate_or_speak(self):
        self.actor.translator = Mock()
        session = self.start_session()
        for result in (self.result('  '), self.result(no_speech_prob=0.9), STTSentinel('flush', None)):
            self.actor.text_in(session.id, result)
        self.actor.translator.assert_not_called()
        self.actor.tts_actr.tts_session_say.remote.assert_not_called()

    def test_translation_error_cleans_up_and_hangs_up(self):
        session = self.start_session()
        self.actor.translator = Mock(side_effect=RuntimeError('translation failed'))
        with self.assertRaisesRegex(RuntimeError, 'translation failed'):
            self.actor.text_in(session.id, self.result())
        self.assertFalse(self.actor.sessions)
        self.actor.stt_actr.stt_session_end.remote.assert_called_once()
        self.actor.tts_actr.tts_session_end.remote.assert_called_once()
        self.actor.sip_actr.sess_term.remote.assert_called_once_with(self.offer.sip_sess_id)

    def test_connect_failure_cleans_partial_start(self):
        self.rtp.rtp_session_connect.remote.side_effect = RuntimeError('RTP closed')
        with self.assertLogs(actor_module.__name__, level='ERROR'):
            with self.assertRaisesRegex(RuntimeError, 'RTP closed'):
                self.start_session()
        self.assertFalse(self.actor.sessions)
        self.actor.stt_actr.stt_session_end.remote.assert_called_once()
        self.actor.tts_actr.tts_session_end.remote.assert_called_once()
        self.actor.sip_actr.sess_term.remote.assert_called_once()

    def test_tts_creation_failure_releases_stt_and_sip(self):
        self.actor.tts_actr.new_tts_session.remote.side_effect = RuntimeError('TTS unavailable')
        with self.assertLogs(actor_module.__name__, level='ERROR'):
            with self.assertRaisesRegex(RuntimeError, 'TTS unavailable'):
                self.start_session()
        self.assertFalse(self.actor.sessions)
        self.actor.stt_actr.stt_session_end.remote.assert_called_once_with(self.stt_id)
        self.actor.tts_actr.tts_session_end.remote.assert_not_called()
        self.actor.sip_actr.sess_term.remote.assert_called_once()

    def test_queued_tts_error_cleans_up(self):
        session = self.start_session()
        self.actor.text_in(session.id, self.result('first'))
        self.actor.text_in(session.id, self.result('second'))
        say = self.actor.tts_actr.tts_session_say.remote
        say.side_effect = RuntimeError('queued TTS failed')
        with self.assertRaisesRegex(RuntimeError, 'queued TTS failed'):
            say.call_args.kwargs['req'].done_cb()
        self.assertFalse(self.actor.sessions)
        self.actor.sip_actr.sess_term.remote.assert_called_once()

    def test_accept_failure_does_not_leave_session(self):
        self.offer.accept.side_effect = KeyError('caller gone')
        with self.assertLogs(actor_module.__name__, level='ERROR'):
            with self.assertRaises(KeyError):
                self.start_session()
        self.assertFalse(self.actor.sessions)
        self.actor.stt_actr.new_stt_session.remote.assert_not_called()
        self.actor.sip_actr.sess_term.remote.assert_not_called()

    def test_one_cleanup_error_does_not_skip_other_services(self):
        session = self.start_session()
        self.actor.stt_actr.stt_session_end.remote.side_effect = RuntimeError('STT gone')
        with self.assertLogs(session_module.__name__, level='ERROR'):
            self.actor.sess_term(session.id, None)
        self.actor.tts_actr.tts_session_end.remote.assert_called_once()
        self.actor.sip_actr.sess_term.remote.assert_called_once()

    def test_tts_error_cleans_up(self):
        session = self.start_session()
        self.actor.tts_actr.tts_session_say.remote.side_effect = RuntimeError('TTS failed')
        with self.assertRaisesRegex(RuntimeError, 'TTS failed'):
            self.actor.text_in(session.id, self.result())
        self.assertFalse(self.actor.sessions)
        self.actor.sip_actr.sess_term.remote.assert_called_once()

    def test_actor_start_translation_needed_vs_bypass(self):
        stt = remote_actor(start=lambda: None)
        tts = remote_actor(start=lambda **kw: None)
        with (
            patch.object(actor_module.InfernSTTActor, 'remote') as stt_factory,
            patch.object(actor_module.InfernTTSActor, 'remote') as tts_factory,
            patch.object(actor_module.IG, 'get_translator') as translator,
            patch.object(actor_module.ray, 'get_runtime_context') as context,
        ):
            stt_factory.return_value = stt
            tts_factory.return_value = tts
            context.return_value.current_actor = self.actor.echo_actr
            self.actor.start(EchoProfile('same', {'stt_lang': 'de', 'tts_lang': 'de'}), self.actor.sip_actr)
            translator.assert_not_called()
            self.assertIsNone(self.actor.translator)
            self.actor.start(EchoProfile('different', {'stt_lang': 'de', 'tts_lang': 'en'}), self.actor.sip_actr)
            translator.assert_called_once_with('de', 'en')
            self.assertIs(self.actor.translator, translator.return_value.translate)
            tts.start.remote.assert_called_with(lang='en', output_sr=8000)


class EchoConfigTests(unittest.TestCase):
    def test_profile_defaults_and_schema(self):
        profile = EchoProfile('default', {})
        self.assertEqual((profile.stt_lang, profile.tts_lang), ('en', 'en'))
        self.assertIsNone(profile.greeting)
        validator = Validator(EchoAppConfig.schema)
        # Absent greeting is valid
        self.assertTrue(validator.validate({'echo': {'profiles': {'default': {}}}}))
        # Empty greeting string is valid (implementation skips it)
        self.assertTrue(validator.validate({'echo': {'profiles': {'default': {'greeting': ''}}}}))
        # Non-empty greeting is valid
        self.assertTrue(validator.validate({'echo': {'profiles': {'default': {'greeting': 'Hello'}}}}))
        # Invalid: unknown option
        self.assertFalse(validator.validate({'echo': {'profiles': {'default': {'llm_prompt': 'not-used'}}}}))
        # Invalid: empty stt_lang
        self.assertFalse(validator.validate({'echo': {'profiles': {'default': {'stt_lang': ''}}}}))
        # Invalid: empty tts_lang
        self.assertFalse(validator.validate({'echo': {'profiles': {'default': {'tts_lang': ''}}}}))
        # Invalid: greeting as non-string
        self.assertFalse(validator.validate({'echo': {'profiles': {'default': {'greeting': 123}}}}))
        # Invalid: old intro key should be rejected
        self.assertFalse(validator.validate({'echo': {'profiles': {'default': {'intro': 'old'}}}}))

    def test_sample_config_loads_and_sip_sink_dispatches(self):
        sip = remote_actor()
        app = remote_actor(start=lambda *args: None, new_sip_session_received=lambda *args: None)
        with (
            patch.object(config_module.InfernSIPActor, 'options') as options,
            patch('Apps.Echo.EchoProfile.EchoActor') as factory,
            patch('Apps.Echo.EchoProfile.ray.get', side_effect=lambda value: value),
            patch('ray.init', side_effect=AssertionError('Ray init forbidden')),
        ):
            options.return_value.remote.return_value = sip
            factory.remote.return_value = app
            conf = config_module.InfernConfig(str(ROOT / 'examples/echo.yaml'))
            self.assertIsInstance(conf.apps['apps/echo/default'], EchoProfile)
            app.start.remote.assert_called_once()
            conf.connectors['sip/incoming'].new_sess_offer('offer')
            app.new_sip_session_received.remote.assert_called_once_with('offer')
            # Profile actor handles are cached.
            self.assertIs(conf.apps['apps/echo/default'].getActor(conf, sip), app)
            factory.remote.assert_called_once()

    def test_actor_can_be_serialized_without_cluster(self):
        from ray import cloudpickle
        # Round-tripping dynamic Ray classes in this process can replace their
        # function globals; export bytes without changing later tests' bindings.
        self.assertGreater(len(cloudpickle.dumps(Actor)), 0)


class EchoGreetingTests(unittest.TestCase):
    """Tests for the optional greeting parameter."""

    def setUp(self):
        # Resolve fake remote results locally; never start a Ray service.
        self.enterContext(patch.object(session_module.ray, 'get', side_effect=lambda value: value))
        self.enterContext(patch.object(session_module.ray, 'init', side_effect=AssertionError('Ray init forbidden')))
        self.actor = Actor()
        self.actor.echo_actr = remote_actor(
            text_in=self.actor.text_in, tts_say_done=self.actor.tts_say_done,
            sess_term=self.actor.sess_term)
        self.stt_id, self.tts_id, self.rtp_id = uuid4(), uuid4(), uuid4()
        self.actor.stt_actr = remote_actor(
            new_stt_session=lambda **kw: self.stt_id,
            stt_session_soundin=lambda **kw: None,
            stt_session_end=lambda sid: None)
        self.actor.tts_actr = remote_actor(
            new_tts_session=lambda: self.tts_id,
            tts_session_start=lambda *args: None,
            tts_session_say=lambda **kw: uuid4(),
            tts_session_end=lambda sid: None)
        self.actor.sip_actr = remote_actor(sess_term=lambda sid: None)
        self.actor.stt_lang = 'en'
        self.actor.tts_lang = 'en'
        self.actor.greeting = None
        self.actor.translator = None
        self.rtp = remote_actor(rtp_session_connect=lambda *args: None,
                                rtp_session_soundout=lambda *args, **kw: None)
        self.offer = SimpleNamespace(sip_sess_id=uuid4(), accept=Mock(
            return_value=(self.rtp, self.rtp_id)))

    def start_session(self):
        self.actor.new_sip_session_received(self.offer)
        return next(iter(self.actor.sessions.values()))

    def result(self, text='Guten Tag', no_speech_prob=0.1):
        chunk = AudioChunk(torch.zeros(160), 16000)
        return STTResult(text, no_speech_prob, STTRequest(chunk, None, 'en'))

    def test_greeting_enqueued_once_on_connect(self):
        # Verify greeting is enqueued once per session on connection.
        self.actor.greeting = 'Hello there'
        session = self.start_session()
        # Greeting should be in the queue after start()
        self.assertEqual(len(session.say_buffer), 1)
        self.assertEqual(session.say_buffer[0].text, 'Hello there')
        # Done callback should be set
        self.assertIsNotNone(session.say_buffer[0].done_cb)

    def test_greeting_sent_once_and_advances_queue(self):
        # Verify greeting is spoken once, then recognized speech follows.
        self.actor.greeting = 'Test greeting'
        session = self.start_session()
        # Greeting is queued
        self.assertEqual(len(session.say_buffer), 1)
        # Simulate greeting completion
        done_cb = self.actor.tts_actr.tts_session_say.remote.call_args.kwargs['req'].done_cb
        done_cb()
        # Greeting removed from queue
        self.assertEqual(len(session.say_buffer), 0)
        # Now send recognized speech
        self.actor.text_in(session.id, self.result('recognized'))
        # Speech should now be queued
        self.assertEqual(len(session.say_buffer), 1)
        self.assertEqual(session.say_buffer[0].text, 'recognized')

    def test_no_greeting_preserves_existing_behavior(self):
        # Verify absent/empty greeting does not enqueue anything.
        self.actor.greeting = None
        session = self.start_session()
        self.assertEqual(len(session.say_buffer), 0)
        # Now send recognized speech - should work normally
        self.actor.text_in(session.id, self.result('hello'))
        self.assertEqual(len(session.say_buffer), 1)
        self.actor.tts_actr.tts_session_say.remote.call_args.kwargs['req'].done_cb()
        self.assertEqual(len(session.say_buffer), 0)

    def test_empty_greeting_preserves_existing_behavior(self):
        # Verify empty string greeting does not enqueue anything.
        self.actor.greeting = ''
        session = self.start_session()
        # Empty greeting should not be enqueued (falsy check)
        self.assertEqual(len(session.say_buffer), 0)

    def test_greeting_with_translation(self):
        # Verify greeting respects tts_lang and translation if needed.
        self.actor.stt_lang = 'de'
        self.actor.tts_lang = 'en'
        self.actor.translator = Mock(side_effect=lambda text: 'EN: ' + text)
        self.actor.greeting = 'Hallo'
        session = self.start_session()
        # Greeting should be enqueued as-is (greeting uses tts_lang, no translation)
        self.assertEqual(len(session.say_buffer), 1)
        self.assertEqual(session.say_buffer[0].text, 'Hallo')

    def test_disconnect_cleans_greeting_queue(self):
        # Verify greeting queue is cleared on disconnect.
        self.actor.greeting = 'Test'
        session = self.start_session()
        self.assertEqual(len(session.say_buffer), 1)
        # Trigger disconnect
        msg = self.offer.accept.call_args.kwargs['msg']
        msg.disc_cb()
        self.assertTrue(session.closed)
        self.assertFalse(session.say_buffer)

    def test_greeting_speech_queue_ordering(self):
        # Verify greeting is spoken before recognized speech.
        self.actor.greeting = 'Greeting first'
        session = self.start_session()
        # Greeting queued
        self.assertEqual(len(session.say_buffer), 1)
        # Queue speech while greeting playing
        self.actor.text_in(session.id, self.result('speech after'))
        self.assertEqual(len(session.say_buffer), 2)
        # Complete greeting
        done_cb = self.actor.tts_actr.tts_session_say.remote.call_args_list[0].kwargs['req'].done_cb
        done_cb()
        # Speech should now be playing
        self.assertEqual(len(session.say_buffer), 1)
        self.assertEqual(session.say_buffer[0].text, 'speech after')


if __name__ == '__main__':
    unittest.main()
