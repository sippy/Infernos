"""Single-leg speech echo application; no LLM or English STT pivot."""

from collections import deque
import logging

import ray

from Cluster.InfernSTTActor import InfernSTTActor
from Cluster.InfernTTSActor import InfernTTSActor
from config.InfernGlobals import InfernGlobals as IG
from Core.Exceptions.InfernSessNotFoundErr import InfernSessNotFoundErr
from .EchoSession import EchoSession

log = logging.getLogger(__name__)


class EchoSessNotFoundErr(InfernSessNotFoundErr):
    pass


@ray.remote(resources={'echo': 1})
class EchoActor:
    def __init__(self):
        self.sessions = {}
        self.tombstones = deque(maxlen=100)
        self.greeting = None

    def start(self, profile, sip_actr):
        self.stt_lang = profile.stt_lang
        self.tts_lang = profile.tts_lang
        self.sip_actr = sip_actr
        self.echo_actr = ray.get_runtime_context().current_actor
        self.translator = None
        if profile.stt_lang != profile.tts_lang:
            self.translator = IG.get_translator(profile.stt_lang, profile.tts_lang).translate
        self.greeting = profile.greeting
        self.stt_actr = InfernSTTActor.remote()
        self.tts_actr = InfernTTSActor.remote()
        ray.get([self.stt_actr.start.remote(),
                 self.tts_actr.start.remote(lang=self.tts_lang, output_sr=8000)])

    def new_sip_session_received(self, new_sess):
        session = EchoSession(self, new_sess)
        self.sessions[session.id] = session
        try:
            session.start(new_sess)
        except Exception:
            log.exception('Could not start Echo session %s', session.id)
            self.sess_term(session.id, None, relaxed=True)
            raise

    def _get_session(self, sess_id):
        try:
            return self.sessions[sess_id]
        except KeyError:
            raise EchoSessNotFoundErr(f'No Echo session with id {sess_id}') from None

    def text_in(self, sess_id, result):
        if sess_id in self.tombstones:
            return
        session = self._get_session(sess_id)
        try:
            session.text_in(result)
        except Exception:
            self.sess_term(sess_id, None, relaxed=True)
            raise

    def tts_say_done(self, sess_id):
        if sess_id not in self.tombstones:
            self._get_session(sess_id).tts_say_done()

    def sess_term(self, sess_id, sip_sess_id, relaxed=False):
        session = self.sessions.pop(sess_id, None)
        if session is None:
            if relaxed or sess_id in self.tombstones:
                return
            raise EchoSessNotFoundErr(f'No Echo session with id {sess_id}')
        self.tombstones.append(sess_id)
        session.sess_term(sip_sess_id)
