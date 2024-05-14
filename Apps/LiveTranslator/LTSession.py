from typing import Tuple, List, Optional, Dict
from functools import partial, lru_cache
from uuid import UUID, uuid4

from ray import ray
from nltk.tokenize import sent_tokenize

from Cluster.InfernRTPActor import InfernRTPActor
from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.RemoteTTSSession import RemoteTTSSession
from Cluster.STTSession import STTRequest, STTResult
from Cluster.TTSSession import TTSRequest
from Core.AudioChunk import AudioChunk, AudioChunkFromURL
from RTP.AudioInput import AudioInput
from Core.T2T.Translator import Translator
from SIP.RemoteSession import RemoteSessionOffer, RemoteSessionAccept, NewRemoteSessionRequest
from Core.AStreamMarkers import ASMarkerNewSent

#from .LTProfile import LTProfile

import pickle
import gzip
from random import choice

@lru_cache(maxsize=4)
def get_top_speakers(lang:str):
    skips = 0
    i = 0
    res = []
    while True:
        try:
            with gzip.open(f'checkpoint/{lang}/speaker.{i}.{lang}.pkl.gz', 'rb') as file:
                res.append(pickle.load(file))
        except FileNotFoundError:
            skips += 1
            if skips > 200: break
        i += 1
    if len(res) == 0:
        return None
    gen = max(r.nres for r in res)
    res = sorted([r for r in res if r.nres == gen], key=lambda r: r.max_error())[:50]
    return tuple(r.speaker_id for r in res)

class VADSignals():
    def __init__(self):
        eng, deng= [AudioChunkFromURL(f'https://github.com/commaai/openpilot/blob/master/selfdrive/assets/sounds/{n}.wav?raw=true') for n in ('engage', 'disengage')]
        eng.track_id = 2
        eng.debug = True
        self.eng = ray.put(eng)
        self.deng = ray.put(deng)

class STTProxy():
    debug = True
    stt_do: callable
    stt_done: callable
    vad_mirror: callable
    def __init__(self, lta:'LTActor', uas:'Sess', stt_done, vad_mirror, direction):
        self.stt_do = partial(lta.stt_actr.stt_session_soundin.remote, sess_id=uas.stt_sess_id)
        self.lang, self.stt_done = uas.stt_lang, stt_done
        self.vad_mirror = vad_mirror
        self.eng = lta.vds.eng
        self.direction = direction

    # This method runs in the context of the inbound RTP Actor
    def __call__(self, chunk:AudioChunk):
        if self.debug:
            dir = 'A' if self.direction == 0 else 'B'
            print(f'STTProxy: VAD({dir}): {len(chunk.audio)=} {chunk.track_id=}')
        #self.vad_mirror(chunk=self.eng)
        def stt_done(result:STTResult, direction=self.direction):
            print(f'STTProxy: {result=}')
            result.direction = direction
            self.stt_done(result=result)
        sreq = STTRequest(chunk, stt_done, self.lang)
        sreq.mode = 'translate'
        self.stt_do(req=sreq)

class TTSProxy():
    debug = False
    tts_consume: callable
    def __init__(self, tts_consume):
        self.tts_consume = tts_consume

    # This method runs in the context of the outbound RTP Actor
    def __call__(self, chunk:AudioChunk):
        if self.debug and isinstance(chunk, ASMarkerNewSent):
            print(f'TTSProxy: ASMarkerNewSent')
        chunk.track_id = 1
        chunk.debug = False
        self.tts_consume(chunk=chunk)

class SessionInfo():
    soundout: callable
    rsess_pause: callable
    rsess_connect: callable
    translator: callable
    get_speaker: callable
    tts_say: callable
    tts_say_done: callable
    def __init__(self, lts:'LTSession', lta:'LTActor', xua:'Sess', yua:'Sess'):
        #lt_actr = ray.get_runtime_context().current_actor
        self.soundout = partial(xua.rtp_actr.rtp_session_soundout.remote, xua.rtp_sess_id)
        vad_cb = self.soundout
        text_cb = partial(lta.lt_actr.text_in.remote, sess_id=lts.id)
        self.tts_say_done = partial(lta.lt_actr.tts_say_done.remote, sess_id=lts.id, direction=xua.direction)
        vad_handler = STTProxy(lta, xua, text_cb, vad_cb, xua.direction)
        self.rsess_pause = partial(xua.rtp_actr.rtp_session_connect.remote, xua.rtp_sess_id,
                                   AudioInput(vad_chunk_in=vad_handler))
        ysoundout = partial(yua.rtp_actr.rtp_session_soundout.remote, yua.rtp_sess_id)
        self.rsess_connect = partial(xua.rtp_actr.rtp_session_connect.remote, xua.rtp_sess_id,
                                     AudioInput(yua.rtp_sess_id, vad_handler))
        self.translator = xua.translator
        self.get_speaker = (lambda: None) if xua.speakers is None else partial(choice, xua.speakers)
        self.sip_sess_term = partial(lta.sip_actr.sess_term.remote, xua.sip_sess_id)
        self.stt_sess_term = partial(lta.stt_actr.stt_session_end.remote, xua.stt_sess_id)
        self.tts_sess_term = xua.tts_sess.end
        self.tts_say = xua.tts_sess.say
        self.tts_soundout = TTSProxy(ysoundout)

    def sess_term(self):
        self.stt_sess_term()
        self.tts_sess_term()

class Sessions():
    info: Tuple[SessionInfo]
    def __init__(self, lts:'LTSession', lta:'LTActor', xua:'Sess', yua:'Sess'):
        self.info = (
            SessionInfo(lts, lta, xua, yua),
            SessionInfo(lts, lta, yua, xua),
            )
        for i, u in zip(self.info, (xua, yua)):
            i.rsess_connect()
            #i.rsess_pause()
            u.tts_sess.start(i.tts_soundout)

class Sess():
    direction: int
    sip_sess_id: UUID
    rtp_sess_id: UUID
    tts_sess: RemoteTTSSession
    stt_sess_id: UUID
    rtp_actr: InfernRTPActor
    tts_actr: InfernTTSActor
    translator: Optional[Translator]
    def __init__(self, lta:'LTActor', direction:int):
        self.direction = direction
        tts_lang, stt_lang = lta.tts_langs[direction], lta.stt_langs[direction]
        self.speakers = get_top_speakers(tts_lang)
        self.tts_lang, self.stt_lang = tts_lang, stt_lang
        self.translator = lta.translators[direction]
        self.tts_sess = RemoteTTSSession(lta.tts_actrs[tts_lang])

def _sess_term(*args, sterm:callable, sess_id:UUID, sip_sess_id:UUID):
    return sterm(sess_id, sip_sess_id, relaxed=True)

class LTSession():
    debug = False
    id: UUID
    alice: Sess
    bob: Sess
    say_buffer: Dict[int, List[TTSRequest]]

    def __init__(self, lta, new_sess:RemoteSessionOffer):

        self.id = uuid4()
        self.say_buffer = {0:[], 1:[]}
        lt_prof: 'LTProfile' = lta.lt_prof
        dest_number = dict(x.split('=', 1) for x in lt_prof.outbount_params.split(';'))['cld']
        #dest_number = '205'
        #dest_number = '601'
        sess_term_alice = partial(_sess_term, sterm=lta.lt_actr.sess_term.remote, sess_id=self.id, sip_sess_id=new_sess.sip_sess_id)
        amsg = RemoteSessionAccept(disc_cb=sess_term_alice, auto_answer=False)
        try:
            rtp_alice = ray.get(new_sess.accept(msg=amsg))
        except KeyError:
            print(f'Failed to accept {new_sess.sip_sess_id=}')
            return
        sess_term_bob = partial(_sess_term, sterm=lta.lt_actr.sess_term.remote, sess_id=self.id, sip_sess_id=None)
        bmsg = NewRemoteSessionRequest(cld=dest_number, sip_prof=lt_prof.outbound_conn, disc_cb=sess_term_bob)
        bmsg.conn_sip_sess_id = new_sess.sip_sess_id
        sip_sess_id_bob = lta.sip_actr.new_sess.remote(msg=bmsg)
        ssess = [lta.stt_actr.new_stt_session.remote(keep_context=True) for _ in lta.stt_langs]

        alice = Sess(lta, 0)
        bob = Sess(lta, 1)

        #alice.tts_sess, bob.tts_sess = [RemoteTTSSession(lta.tts_actrs[lang]) for lang in lta.tts_langs]

        alice.sip_sess_id = new_sess.sip_sess_id
        alice.rtp_actr, alice.rtp_sess_id = rtp_alice
        bob.sip_sess_id, bob.rtp_actr, bob.rtp_sess_id = ray.get(sip_sess_id_bob)
        alice.stt_sess_id, bob.stt_sess_id = ray.get(ssess)
        self.fabric = Sessions(self, lta, alice, bob)
        if self.debug: print(f'{alice=} {bob=} {self.fabric=}')
        self.alice, self.bob = alice, bob

    def sess_term(self, sip_sess_id):
        for i in self.fabric.info:
            i.sess_term()
        if sip_sess_id == self.alice.sip_sess_id:
            self.fabric.info[1].sip_sess_term()
        else:
            self.fabric.info[0].sip_sess_term()

    def text_in(self, result:STTResult):
        sdir = 'A->B' if result.direction == 0 else 'B->A'
        print(f'STT: {sdir} "{result.text=}" {result.no_speech_prob=}')
        nsp = result.no_speech_prob
        if nsp > 0.5: return
        sinfo = self.fabric.info[result.direction]
        text = sinfo.translator(result.text)
        speaker_id = sinfo.get_speaker()
        #sinfo.rsess_pause()
        print(f'TTS: {sdir} "{text=}" {speaker_id=}')
        text = sent_tokenize(text)
        out_sents = [text.pop(0),]
        for t in text:
            if len(out_sents[-1]) + len(t) < 128 or out_sents[-1].endswith(' i.e.'):
                out_sents[-1] += ' ' + t
            else:
                out_sents.append(t)

        print(f'TTS split: "{out_sents=}" {[len(t) for t in out_sents]=}')
        tts_req = ray.put(TTSRequest(out_sents, speaker_id=speaker_id, done_cb=sinfo.tts_say_done))
        self.say_buffer[result.direction].append(tts_req)
        if len(self.say_buffer[result.direction]) > 1:
            return
        sinfo.tts_say(tts_req)
        return

    def tts_say_done(self, direction:int):
        if self.debug: print(f'tts_say_done({direction=})')
        tbuf = self.say_buffer[direction]
        tbuf.pop(0)
        if len(tbuf) > 0:
            self.fabric.info[direction].tts_say(tbuf[0])
            return
