"""Offline RTP startup tests: real ingest threads, fake VAD, no Ray cluster."""

import importlib
from threading import Event, Thread
import unittest
import torch
from unittest.mock import Mock, patch

ingest_module = importlib.import_module('RTP.InfernRTPIngest')
actor_module = importlib.import_module('Cluster.InfernRTPActor')
Ingest = ingest_module.InfernRTPIngest
# Exercise the actual actor methods locally without allocating a Ray actor.
Actor = actor_module.InfernRTPActor.__ray_metadata__.modified_class


class IngestStartupTests(unittest.TestCase):
    def stop_worker(self, worker):
        if worker.is_alive():
            worker.stop()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())

    def test_constructor_failure_is_raised_and_worker_joined(self):
        error = RuntimeError('unsupported device')
        worker = Ingest('mps')
        with patch.object(ingest_module, 'SileroVADWorker', side_effect=error):
            with self.assertRaises(RuntimeError) as result:
                worker.start()
        self.assertIs(result.exception, error)
        self.assertFalse(worker.is_alive())

    def test_vad_start_failure_is_raised_and_worker_joined(self):
        error = RuntimeError('VAD start failed')
        vad = Mock()
        vad.start.side_effect = error
        worker = Ingest('cpu')
        with patch.object(ingest_module, 'SileroVADWorker', return_value=vad):
            with self.assertRaises(RuntimeError) as result:
                worker.start()
        self.assertIs(result.exception, error)
        self.assertFalse(worker.is_alive())

    def test_start_waits_for_readiness_and_stop_cleans_up(self):
        entered, release, finished = Event(), Event(), Event()
        errors = []
        vad = Mock()

        def start_vad():
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError('test readiness gate timed out')

        vad.start.side_effect = start_vad
        worker = Ingest('cpu')

        def start_worker():
            try:
                worker.start()
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        caller = Thread(target=start_worker, daemon=True)
        with patch.object(ingest_module, 'SileroVADWorker', return_value=vad):
            caller.start()
            try:
                self.assertTrue(entered.wait(timeout=2))
                self.assertFalse(finished.is_set(), 'start returned before VAD readiness')
                release.set()
                self.assertTrue(finished.wait(timeout=2))
                self.assertEqual(errors, [])
                self.assertTrue(worker.is_alive())
                self.assertTrue(hasattr(worker, 'pkt_queue'))
            finally:
                release.set()
                caller.join(timeout=4)
                if worker.ident is not None:
                    self.stop_worker(worker)
        vad.start.assert_called_once()
        vad.stop.assert_called_once()


class ActorBackendTests(unittest.TestCase):
    def setUp(self):
        self.attempts = []
        self.workers = []
        self.failed_devices = set()
        self.vads = []
        self.actor = Actor.__new__(Actor)

        def make_worker(device):
            worker = Ingest(device)
            self.workers.append(worker)
            return worker

        def make_vad(device):
            self.attempts.append(device)
            if device in self.failed_devices:
                raise RuntimeError(f'{device} initialization failed')
            vad = Mock()
            self.vads.append(vad)
            return vad

        self.addCleanup(self.stop_workers)
        self.enterContext(patch.object(actor_module, 'InfernRTPIngest', side_effect=make_worker))
        self.enterContext(patch.object(ingest_module, 'SileroVADWorker', side_effect=make_vad))
        self.mps_built = self.enterContext(patch.object(torch.backends.mps, 'is_built', return_value=False))
        self.mps_available = self.enterContext(patch.object(torch.backends.mps, 'is_available', return_value=False))
        self.cuda_available = self.enterContext(patch.object(torch.cuda, 'is_available', return_value=False))

    def stop_workers(self):
        for worker in self.workers:
            if worker.is_alive():
                worker.stop()
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())
        for vad in self.vads:
            vad.stop.assert_called_once()

    def test_unavailable_backends_are_skipped(self):
        self.actor.start()
        self.assertEqual(self.attempts, ['cpu'])
        self.assertEqual(self.actor.device, 'cpu')
        self.assertTrue(self.actor.ring.is_alive())
        self.mps_available.assert_not_called()

    def test_built_but_unavailable_mps_is_skipped(self):
        self.mps_built.return_value = True
        self.actor.start()
        self.assertEqual(self.attempts, ['cpu'])
        self.mps_available.assert_called_once()

    def test_available_mps_is_used(self):
        self.mps_built.return_value = True
        self.mps_available.return_value = True
        self.actor.start()
        self.assertEqual(self.attempts, ['mps'])
        self.assertEqual(self.actor.device, 'mps')
        self.cuda_available.assert_not_called()

    def test_available_cuda_is_used(self):
        self.cuda_available.return_value = True
        self.actor.start()
        self.assertEqual(self.attempts, ['cuda'])
        self.assertEqual(self.actor.device, 'cuda')

    def test_cuda_initialization_failure_falls_back_to_cpu(self):
        self.cuda_available.return_value = True
        self.failed_devices.add('cuda')
        self.actor.start()
        self.assertEqual(self.attempts, ['cuda', 'cpu'])
        self.assertEqual(self.actor.device, 'cpu')
        self.assertFalse(self.workers[0].is_alive())
        self.assertIs(self.actor.ring, self.workers[1])

    def test_all_devices_failing_raises(self):
        self.failed_devices.add('cpu')
        with self.assertRaisesRegex(RuntimeError, 'No suitable device found'):
            self.actor.start()
        self.assertEqual(self.attempts, ['cpu'])
        self.assertFalse(hasattr(self.actor, 'ring'))
        self.assertFalse(self.workers[0].is_alive())


class ActorSerializationTests(unittest.TestCase):
    def test_rtp_actor_can_be_exported(self):
        from ray import cloudpickle
        actor_class = actor_module.InfernRTPActor.__ray_metadata__.modified_class
        restored = cloudpickle.loads(cloudpickle.dumps(actor_class))
        self.assertEqual(restored.__name__, actor_class.__name__)

    def test_sip_actor_can_be_exported(self):
        from ray import cloudpickle
        sip_module = importlib.import_module('Cluster.InfernSIPActor')
        actor_class = sip_module.InfernSIPActor.__ray_metadata__.modified_class
        restored = cloudpickle.loads(cloudpickle.dumps(actor_class))
        self.assertEqual(restored.__name__, actor_class.__name__)


if __name__ == '__main__':
    unittest.main()
