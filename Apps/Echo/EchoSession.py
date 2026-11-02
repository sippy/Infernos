"""RTP VAD -> native-language STT -> optional translation -> TTS -> RTP."""

from collections import deque
from functools import partial
import logging
from uuid import uuid4

import ray

from Cluster.RemoteTTSSession import RemoteTTSSession
from Cluster.STTSession import STTRequest, STTResult
from Cluster.TTSSession import TTSRequest
from RTP.AudioInput import AudioInput
from SIP.RemoteSession import RemoteSessionAccept

log = logging.getLogger(__name__)


def _sess_term(*args, sterm, sess_id, sip_sess_id):
    """Discard sippy disconnect event arguments before invoking the app actor."""
    return sterm(sess_id=sess_id, sip_sess_id=sip_sess_id, relaxed=True)


class STTProxy:
    """Callable installed explicitly in AudioInput.vad_chunk_in.

    AudioInput.__init__ assigns callback fields; subclass methods with those
    names would be shadowed by the instance fields.
    """

    def __init__(self, actor, lang, sess_id, text_cb):
        self.soundin = partial(actor.stt_session_soundin.remote, sess_id=sess_id)
        self.lang, self.text_cb = lang, text_cb

    def __call__(self, chunk):
        req = STTRequest(chunk, self.text_cb, self.lang)
        req.mode = 'transcribe'
        return self.soundin(req=req)


class TTSProxy:
    def __init__(self, soundout):
        self.soundout = soundout

    def __call__(self, chunk):
        chunk.track_id = 1
        chunk.debug = False
        return self.soundout(chunk=chunk)


class EchoSession:
    def __init__(self, actor, offer):
        self.id = uuid4()
        self.actor = actor
        self.sip_sess_id = offer.sip_sess_id
        self.accepted = False
        self.closed = False
        self.stt_sess_id = None
        self.tts_sess = None
        self.say_buffer = deque()

    def start(self, offer):
        disc_cb = partial(_sess_term, sterm=self.actor.echo_actr.sess_term.remote,
                          sess_id=self.id, sip_sess_id=self.sip_sess_id)
        msg = RemoteSessionAccept(disc_cb=disc_cb, auto_answer=True)
        self.rtp_actr, self.rtp_sess_id = ray.get(offer.accept(msg=msg))
        self.accepted = True
        self.stt_sess_id = ray.get(self.actor.stt_actr.new_stt_session.remote(keep_context=True))
        self.tts_sess = RemoteTTSSession(self.actor.tts_actr)
        soundout = partial(self.rtp_actr.rtp_session_soundout.remote, self.rtp_sess_id)
        self.tts_sess.start(TTSProxy(soundout))
        text_cb = partial(self.actor.echo_actr.text_in.remote, sess_id=self.id)
        vad = STTProxy(self.actor.stt_actr, self.actor.stt_lang, self.stt_sess_id, text_cb)
        ray.get(self.rtp_actr.rtp_session_connect.remote(
            self.rtp_sess_id, AudioInput(vad_chunk_in=vad)))
        # Enqueue greeting text once per session if configured.
        if self.actor.greeting:
            done_cb = partial(self.actor.echo_actr.tts_say_done.remote, sess_id=self.id)
            self.say_buffer.append(TTSRequest(self.actor.greeting, done_cb=done_cb))
            if len(self.say_buffer) == 1:
                ray.get(self.tts_sess.say(self.say_buffer[0]))

    def text_in(self, result):
        if self.closed or not isinstance(result, STTResult):
            return
        if result.no_speech_prob > STTRequest.max_ns_prob or not result.text.strip():
            return
        text = result.text
        if self.actor.translator is not None:
            text = self.actor.translator(text)
        if not text.strip():
            return
        done_cb = partial(self.actor.echo_actr.tts_say_done.remote, sess_id=self.id)
        # A string needs no NLTK sentence tokenizer or downloaded tokenizer data.
        self.say_buffer.append(TTSRequest(text, done_cb=done_cb))
        if len(self.say_buffer) == 1:
            ray.get(self.tts_sess.say(self.say_buffer[0]))

    def tts_say_done(self):
        if self.closed or not self.say_buffer:
            return
        self.say_buffer.popleft()
        if self.say_buffer:
            try:
                ray.get(self.tts_sess.say(self.say_buffer[0]))
            except Exception:
                self.actor.sess_term(self.id, None, relaxed=True)
                raise

    def sess_term(self, sip_sess_id):
        if self.closed:
            return
        self.closed = True
        self.say_buffer.clear()
        # SIP owns RTP teardown. An app error hangs up the accepted SIP leg;
        # a normal SIP disconnect has already initiated that teardown.
        actions = []
        if self.stt_sess_id is not None:
            actions.append(lambda: ray.get(self.actor.stt_actr.stt_session_end.remote(self.stt_sess_id)))
        if self.tts_sess is not None:
            actions.append(self.tts_sess.end)
        if self.accepted and sip_sess_id != self.sip_sess_id:
            actions.append(lambda: ray.get(self.actor.sip_actr.sess_term.remote(self.sip_sess_id)))
        for action in actions:
            try:
                action()
            except Exception:
                # One missing/already closed service must not leak the others.
                log.exception('Echo session %s cleanup failed', self.id)
