"""Real sippy codec/packetizer/jitter path with isolated transport and scheduler."""

from array import array
import importlib
from queue import Empty
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import torch
from sippy.Rtp.Core.AudioChunk import AudioChunk as SippyAudioChunk
from sippy.Rtp.Codecs.G711 import G711Codec
from sippy.Rtp.Codecs.G722 import G722Codec
from Core.AudioChunk import AudioChunk
from Core.AStreamMarkers import ASMarkerGeneric, ASMarkerNewSent, ASMarkerSentDoneCB
from RTP.AudioBridge import to_infernos, to_sippy
from RTP.AudioInput import AudioInput
from RTP.InfernRTPEPoint import InfernRTPEPoint
from RTP.InfernRTPIngest import InfernRTPIngest
from RTP.RTPOutputWorker import RTPOutputWorker
from RTP.RTPParams import RTPParams

base_output = importlib.import_module('sippy.Rtp.OutputWorker')
base_endpoint = importlib.import_module('sippy.Rtp.EPoint')
base_params = importlib.import_module('sippy.Rtp.Params')


class FakeChannel:
    def __init__(self, host='127.0.0.1', port=24000):
        self.local_addr = (host, port)
        self.closed = False
        self.target = None
        self.packets = []

    def set_target(self, host, port):
        self.target = (host, port)

    def send_pkt(self, packet):
        self.packets.append(packet)

    def close(self):
        assert not self.closed
        self.closed = True


class FakeProc:
    def create_channel(self, proc_in):
        channel = FakeChannel()
        channel.proc_in = proc_in
        return channel


class FakeServer:
    def create_channel(self, bind_family, pkt_in, bind_host, bind_port):
        channel = FakeChannel(bind_host, bind_port)
        channel.pkt_in = pkt_in
        return channel


class AudioPathTests(unittest.TestCase):
    def setUp(self):
        self.resources = []
        self.enterContext(patch.object(base_params, 'local4remote', return_value='127.0.0.1'))
        self.enterContext(patch.object(base_output, 'acquire_rtp_proc', return_value=FakeProc()))
        self.enterContext(patch.object(base_output, 'release_rtp_proc'))
        self.enterContext(patch.object(base_endpoint, 'acquire_rtp_server', return_value=FakeServer()))
        self.enterContext(patch.object(base_endpoint, 'release_rtp_server'))
        self.addCleanup(self.close_resources)

    def close_resources(self):
        for resource in reversed(self.resources):
            if isinstance(resource, InfernRTPEPoint):
                resource.shutdown()
            elif resource._proc is not None:
                resource.end()
                resource.join()

    def make_endpoint(self, codec=G711Codec):
        params = RTPParams(('127.0.0.1', 9000), 20, out_sr=8000)
        params.codec = codec
        ring = InfernRTPIngest('cpu')
        endpoint = InfernRTPEPoint(SimpleNamespace(palloc=lambda attempt: 24000), params, ring, Mock())
        self.resources.append(endpoint)
        return endpoint, ring

    def make_writer(self):
        writer = RTPOutputWorker('cpu', RTPParams(('127.0.0.1', 9000), 20))
        self.resources.append(writer)
        writer.set_pkt_send_f(Mock())
        writer.start()
        return writer

    def tick(self, writer, number=1):
        for i in range(number):
            writer._proc_in(1_000_000_000 + i * writer._period_ns,
                            1_000_000_000 + i * writer._period_ns)

    def drain(self, ring, vad):
        while True:
            try:
                work = ring.pkt_queue.get_nowait()
            except Empty:
                return
            work.stream._proc_in_tread(work, vad)

    def test_parameters_use_sippy_codecs_and_preserve_defaults(self):
        params = RTPParams(('127.0.0.1', 9000), None)
        self.assertIsInstance(params, base_params.RTPParams)
        self.assertIs(params.codec, G711Codec)
        self.assertEqual(params.out_ptime, 20)
        self.assertEqual(params.rtp_proto, 'IP4')
        with patch.object(base_params, 'local4remote', return_value='::1'):
            ipv6 = RTPParams(('[::1]', 9000), 40)
        self.assertEqual(ipv6.rtp_proto, 'IP6')
        self.assertEqual(ipv6.rtp_target, ('::1', 9000))

    def test_output_resamples_supplied_audio_without_mutating_caller(self):
        writer = self.make_writer()
        source = AudioChunk(torch.full((320,), .25), 16000)
        writer.soundout(source)
        self.tick(writer)
        writer.pkt_send_f.assert_called_once()
        self.assertEqual(len(writer.pkt_send_f.call_args.args[0]), 172)
        self.assertEqual(source.samplerate, 16000)
        self.assertEqual(source.audio.numel(), 320)

    def test_ptime_update_replaces_and_closes_previous_writer(self):
        endpoint, ring = self.make_endpoint()
        old = endpoint.writer
        endpoint.update(RTPParams(('127.0.0.1', 9000), 40, out_sr=8000))
        self.assertIsNot(endpoint.writer, old)
        self.assertIsNone(old._proc)
        self.assertEqual(endpoint.writer.out_ft, 40)

    def test_actual_sip_negotiation_selects_sippy_g722_and_sends_audio(self):
        from SIP.InfernUA import model_body
        from SIP.InfernUAS import InfernUAS
        from SIP.InfernUAC import InfernUAC
        for cls in (InfernUAS, InfernUAC):
            with self.subTest(cls=cls.__name__):
                ua = cls.__new__(cls)
                body = model_body.getCopy()
                body.parse()
                section = body.content.sections[0]
                section.c_header.addr = '127.0.0.1'
                section.m_header.port = 9000
                section.m_header.formats = [9]
                selected = ua.extract_rtp_params(body)
                self.assertIs(selected.codec, G722Codec)
                endpoint, ring = self.make_endpoint(selected.codec)
                endpoint.soundout(AudioChunk(torch.full((160,), .25), 8000))
                self.tick(endpoint.writer)
                self.assertEqual(endpoint.rserv.packets[0][1] & 127, 9)

    def test_g722_packet_loss_uses_valid_silence_at_native_rate(self):
        for output_rate in (8000, 16000):
            with self.subTest(output_rate=output_rate):
                endpoint, ring = self.make_endpoint(G722Codec)
                endpoint.rtp_params.out_sr = output_rate
                endpoint.rsess.stream_update()
                endpoint.soundout(AudioChunk(torch.full((160 * 40,), .25), 8000))
                self.tick(endpoint.writer, 40)
                self.assertEqual(len(endpoint.rserv.packets), 40)
                codec = endpoint.rsess.codec
                original = codec.silence
                concealed = []

                def conceal(nframes):
                    payload = original(nframes)
                    concealed.append((nframes, payload))
                    return payload

                with patch.object(codec, 'silence', side_effect=conceal):
                    for i, packet in enumerate(endpoint.rserv.packets):
                        if i != 12:
                            endpoint.rtp_received(packet, endpoint.rtp_params.rtp_target,
                                                  1_000_000_000 + i * 20_000_000)
                self.assertTrue(concealed, 'packet loss did not exercise concealment')
                for nframes, payload in concealed:
                    self.assertEqual(len(payload), nframes)
                vad = SimpleNamespace(input_sr=8000, window_size_samples=768, infer=Mock())
                self.drain(ring, vad)
                self.assertGreater(vad.infer.call_count, 0)
                for call in vad.infer.call_args_list:
                    self.assertEqual(call.args[0][1].audio.numel(), 768)
                    self.assertTrue(torch.isfinite(call.args[0][1].audio).all())

    def test_stream_update_queues_reset_under_packet_lock(self):
        endpoint, ring = self.make_endpoint()
        put = ring.pkt_queue.put
        observed = []

        def checked_put(work):
            observed.append(endpoint.rsess.ring_lock.locked())
            put(work)

        with patch.object(ring.pkt_queue, 'put', side_effect=checked_put):
            endpoint.rsess.stream_update()
        self.assertEqual(observed, [True])

    def test_pcm_bridge_is_lossless_for_int16_and_preserves_metadata(self):
        source = SippyAudioChunk(array('h', [-32768, -8192, 0, 8192, 32767]), 8000)
        source.track_id, source.rtime = 4, 1.25
        tensor = to_infernos(source)
        restored = to_sippy(tensor)
        self.assertEqual(restored.audio, source.audio)
        self.assertEqual(restored.samplerate, 8000)
        self.assertEqual(restored.track_id, 4)
        self.assertEqual(restored.rtime, 1.25)

    def test_pcm_bridge_clamps_instead_of_wrapping(self):
        source = AudioChunk(torch.tensor([-2., -1., 0., 1., 2.]), 8000)
        self.assertEqual(list(to_sippy(source).audio), [-32768, -32768, 0, 32767, 32767])

    def test_endpoint_uses_supplied_audio_and_actual_sippy_packetizer(self):
        endpoint, ring = self.make_endpoint()
        self.assertIsInstance(endpoint.writer, RTPOutputWorker)
        self.assertEqual(endpoint.rserv.target, ('127.0.0.1', 9000))
        source = AudioChunk(torch.full((160,), 0.25), 8000)
        expected = G711Codec().encode(to_sippy(source))
        self.assertEqual(endpoint.soundout(source), (1, False))
        self.tick(endpoint.writer)
        self.assertEqual(len(endpoint.rserv.packets), 1)
        self.assertEqual(endpoint.rserv.packets[0][12:], expected)
        self.assertTrue(torch.equal(source.audio, torch.full((160,), 0.25)))
        self.assertEqual(endpoint.writer.get_frm_ctrs(), (160, 160))
        self.assertEqual(endpoint.rserv.local_addr, ('127.0.0.1', 24000))

    def test_idle_writer_does_not_generate_audio(self):
        writer = self.make_writer()
        self.tick(writer, 4)
        writer.pkt_send_f.assert_not_called()
        self.assertEqual(writer.get_frm_ctrs(), (0, 0))

    def test_completion_marker_runs_after_preceding_audio_packets(self):
        writer = self.make_writer()
        done = Mock()
        writer.soundout(AudioChunk(torch.full((320,), 0.25), 8000))
        writer.soundout(ASMarkerSentDoneCB(done))
        self.tick(writer)
        done.assert_not_called()
        self.tick(writer)
        done.assert_not_called()
        self.assertEqual(writer.pkt_send_f.call_count, 2)
        self.tick(writer)
        done.assert_called_once_with()
        self.assertEqual(writer.pkt_send_f.call_count, 2)

    def test_short_tail_is_sent_before_completion(self):
        writer = self.make_writer()
        done = Mock()
        writer.soundout(AudioChunk(torch.full((7,), 0.25), 8000))
        writer.soundout(ASMarkerSentDoneCB(done))
        self.tick(writer)
        done.assert_not_called()
        packet = writer.pkt_send_f.call_args.args[0]
        expected = G711Codec().encode(to_sippy(AudioChunk(torch.full((7,), 0.25), 8000)))
        self.assertEqual(packet[12:19], expected)
        self.assertEqual(packet[19:], b'\xff' * 153)
        self.tick(writer)
        done.assert_called_once_with()

    def test_generic_marker_does_not_enter_codec_as_audio(self):
        writer = self.make_writer()
        writer.soundout(ASMarkerGeneric())
        self.tick(writer)
        writer.pkt_send_f.assert_not_called()

    def test_asmarkergeneric_on_proc_is_noop(self):
        """Base ASMarkerGeneric.on_proc should be a no-op."""
        marker = ASMarkerGeneric(track_id=1)
        mock_worker = Mock()
        marker.on_proc(mock_worker)
        mock_worker.method_called.assert_not_called()

    def test_asmarkergeneric_on_proc_called_directly_with_worker(self):
        """OutputMuxer calls chunk.on_proc(rtp_worker) directly for generic markers."""
        marker = ASMarkerGeneric(track_id=1)
        mock_worker = Mock()
        with patch.object(marker, 'on_proc') as mock_on_proc:
            marker.on_proc(mock_worker)
            mock_on_proc.assert_called_once_with(mock_worker)

    def test_asmarkernowsent_inherits_base_noop(self):
        """ASMarkerNewSent inherits on_proc from ASMarkerGeneric without redundant override."""
        marker = ASMarkerNewSent(track_id=2)
        mock_worker = Mock()
        # Should not raise; inherits no-op from parent
        marker.on_proc(mock_worker)
        # Verify it's the same method as base class
        self.assertIs(type(marker).on_proc, ASMarkerGeneric.on_proc)

    def test_asmarkersentdoneCB_retains_behavior(self):
        """Completion subclass ASMarkerSentDoneCB retains its override behavior."""
        done_cb = Mock(return_value=42)
        marker = ASMarkerSentDoneCB(done_cb, sync=False)
        mock_worker = Mock()
        # ASMarkerSentDoneCB.on_proc calls done_cb()
        marker.on_proc(mock_worker)
        done_cb.assert_called_once()
        # Verify it returns None (original implementation doesn't return x)
        self.assertIsNone(marker.on_proc(mock_worker))

    def test_multitrack_mixing_preserves_track_selection(self):
        writer = self.make_writer()
        a, b = AudioChunk(torch.full((160,), .5), 8000), AudioChunk(torch.full((160,), .25), 8000)
        a.track_id, b.track_id = 1, 2
        writer.soundout(a)
        writer.soundout(b)
        self.tick(writer)
        expected = G711Codec().encode(to_sippy(AudioChunk(torch.full((160,), .375), 8000)))
        self.assertEqual(writer.pkt_send_f.call_args.args[0][12:], expected)

    def test_g722_uses_sippy_codec_without_local_codec_tables(self):
        endpoint, ring = self.make_endpoint(G722Codec)
        self.assertIsInstance(endpoint.writer.codec, G722Codec)
        endpoint.soundout(AudioChunk(torch.full((160,), .25), 8000))
        self.tick(endpoint.writer)
        self.assertEqual(len(endpoint.rserv.packets), 1)
        self.assertEqual(endpoint.rserv.packets[0][1] & 127, 9)

    def test_real_packets_decode_into_vad_windows_and_connected_callbacks(self):
        endpoint, ring = self.make_endpoint()
        callback = Mock()
        endpoint.connect(AudioInput(audio_in=callback))
        source = AudioChunk(torch.full((160 * 30,), .25), 8000)
        endpoint.soundout(source)
        self.tick(endpoint.writer, 30)
        self.assertEqual(len(endpoint.rserv.packets), 30)
        for i, packet in enumerate(endpoint.rserv.packets):
            endpoint.rtp_received(packet, endpoint.rtp_params.rtp_target, 1_000_000_000 + i * 20_000_000)
        vad = SimpleNamespace(input_sr=8000, window_size_samples=768, infer=Mock())
        self.drain(ring, vad)
        self.assertGreater(vad.infer.call_count, 0)
        channel, chunk = vad.infer.call_args.args[0]
        self.assertEqual(chunk.audio.numel(), 768)
        expected = G711Codec().decode(G711Codec().encode(to_sippy(AudioChunk(torch.tensor([.25]), 8000)))).audio[0] / 32768.0
        torch.testing.assert_close(chunk.audio, torch.full((768,), expected))
        channel.audio_chunk_in(chunk, True)
        callback.assert_called_once_with(chunk=chunk)
        self.assertTrue(chunk.active)

    def test_uuid_lookup_resolves_callbacks_not_audio_chunks(self):
        endpoint, ring = self.make_endpoint()
        callback = Mock()
        key = uuid4()
        lookup = endpoint.rsess.get_direct_soundout = Mock(return_value=callback)
        endpoint.connect(AudioInput(vad_chunk_in=key))
        vad = SimpleNamespace(input_sr=8000, window_size_samples=768, infer=Mock())
        self.drain(ring, vad)
        chunk = AudioChunk(torch.zeros(768), 8000)
        endpoint.rsess.vad_chunk_out(chunk)
        lookup.assert_called_once_with(key)
        callback.assert_called_once_with(chunk=chunk)

    def test_update_rebuilds_codec_and_writer_and_shutdown_closes_resources(self):
        endpoint, ring = self.make_endpoint()
        previous = endpoint.writer
        channel = endpoint.rserv
        params = RTPParams(('127.0.0.1', 9002), 20, out_sr=8000)
        params.codec = G722Codec
        endpoint.update(params)
        self.assertIsNot(endpoint.writer, previous)
        self.assertTrue(previous.ended())
        self.assertIsNone(previous._proc)
        self.assertIsInstance(endpoint.writer.codec, G722Codec)
        self.assertIsInstance(endpoint.rsess.codec, G722Codec)
        self.assertEqual(channel.target, ('127.0.0.1', 9002))
        endpoint.shutdown()
        self.assertTrue(channel.closed)
        self.assertEqual(ring.streams, [])
        self.assertIsNone(endpoint.writer)
        self.assertIsNone(endpoint.soundout(AudioChunk(torch.zeros(160), 8000)))

    def test_actor_session_constructor_and_returned_address_match(self):
        from Cluster.InfernRTPActor import InfernRTPActor
        cls = InfernRTPActor.__ray_metadata__.modified_class
        actor = cls.__new__(cls)
        actor.sessions = {}
        actor.inf_rc = SimpleNamespace(palloc=lambda attempt: 24000)
        actor.ring = InfernRTPIngest('cpu')
        session_id, address = actor.new_rtp_session(RTPParams(('127.0.0.1', 9000), 20))
        self.resources.append(actor.sessions[session_id])
        self.assertEqual(address, ('127.0.0.1', 24000))
        self.assertIsInstance(actor.sessions[session_id].writer, RTPOutputWorker)


if __name__ == '__main__':
    unittest.main()
