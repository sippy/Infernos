from typing import Tuple, List, Optional, Dict, Union
from uuid import UUID, uuid4
from functools import partial
import ray

from nltk.tokenize import sent_tokenize

from Cluster.TTSSession import TTSRequest
from Cluster.STTSession import STTRequest, STTResult, STTSentinel
from Cluster.LLMSession import LLMRequest, LLMResult, LLMSessionParams
from Cluster.RemoteTTSSession import RemoteTTSSession
from Cluster.InfernRTPActor import InfernRTPActor, RTPSessNotFoundErr
from Core.T2T.NumbersToWords import NumbersToWords
from RTP.AudioInput import AudioInput
from SIP.RemoteSession import RemoteSessionOffer, RemoteSessionAccept
from Core.T2T.Translator import Translator
from Core.AudioChunk import AudioChunk
from ..LiveTranslator.LTSession import _sess_term, TTSProxy

class STTProxy(AudioInput):
    from time import monotonic
    last_chunk_time: Optional[float] = None
    debug = True
    stt_do: callable
    stt_done: callable
    def __init__(self, stt_actr, stt_lang, stt_sess_id, stt_done):
        self.stt_do = partial(stt_actr.stt_session_soundin.remote, sess_id=stt_sess_id)
        self.lang, self.stt_done = stt_lang, stt_done

    def audio_in(self, chunk:AudioChunk):
        if self.last_chunk_time is None:
            return
        if chunk.active:
            self.last_chunk_time = None
            return
        if self.monotonic() - self.last_chunk_time < 2.0:
            return
        def stt_done(result:STTSentinel):
            print(f'STTProxy: {result=}')
            self.stt_done(result=result)
        self.last_chunk_time = None
        sreq = STTSentinel('flush', stt_done)
        self.stt_do(req=sreq)

    # This method runs in the context of the inbound RTP Actor
    def vad_chunk_in(self, chunk:AudioChunk):
        self.last_chunk_time = self.monotonic()
        if self.debug:
            print(f'STTProxy: VAD: {len(chunk.audio)=} {chunk.track_id=}')
        def stt_done(result:STTResult):
            print(f'STTProxy: {result=}')
            self.stt_done(result=result)
        sreq = STTRequest(chunk, stt_done, self.lang)
        sreq.mode = 'translate'
        self.stt_do(req=sreq)

class AIASession():
    debug = False
    id: UUID
    stt_sess_id: UUID
    rtp_sess_id: UUID
    llm_sess_id: UUID
    last_llm_req_id: UUID
    rtp_actr: InfernRTPActor
    tts_sess: RemoteTTSSession
    say_buffer: List[TTSRequest]
    translator: Optional[Translator]
    stt_sess_term: callable
    text_in_buffer: List[str]
    saying: UUID

    def __init__(self, aiaa:'AIAActor', new_sess:RemoteSessionOffer, llm_prompt:str):
        self.id = uuid4()
        self.say_buffer = []
        sess_term_alice = partial(_sess_term, sterm=aiaa.aia_actr.sess_term.remote, sess_id=self.id, sip_sess_id=new_sess.sip_sess_id)
        self.tts_say_done_cb = partial(aiaa.aia_actr.tts_say_done.remote, sess_id=self.id)
        amsg = RemoteSessionAccept(disc_cb=sess_term_alice, auto_answer=True)
        try:
            rtp_alice = ray.get(new_sess.accept(msg=amsg))
        except KeyError:
            print(f'Failed to accept {new_sess.sip_sess_id=}')
            return
        self.rtp_actr, self.rtp_sess_id = rtp_alice
        stt_sess = aiaa.stt_actr.new_stt_session.remote(keep_context=True)
        llmp = LLMSessionParams(llm_prompt)
        llm_sess = aiaa.llm_actr.new_llm_session.remote(llmp)
        self.tts_sess = RemoteTTSSession(aiaa.tts_actr)
        self.stt_sess_id, self.llm_sess_id = ray.get([stt_sess, llm_sess])
        self.stt_sess_term = partial(aiaa.stt_actr.stt_session_end.remote, self.stt_sess_id)
        self.llm_sess_term = partial(aiaa.llm_actr.llm_session_end.remote, self.llm_sess_id)
        self.translator = aiaa.translator
        text_cb = partial(aiaa.aia_actr.text_in.remote, sess_id=self.id)
        vad_handler = STTProxy(aiaa.stt_actr, aiaa.stt_lang, self.stt_sess_id, text_cb)
        try:
            ray.get(self.rtp_actr.rtp_session_connect.remote(self.rtp_sess_id, vad_handler))
        except RTPSessNotFoundErr:
            print(f'RTPSessNotFoundErr: {self.rtp_sess_id=}')
            sess_term_alice()
            return
        soundout = partial(self.rtp_actr.rtp_session_soundout.remote, self.rtp_sess_id)
        tts_soundout = TTSProxy(soundout)
        self.tts_sess.start(tts_soundout)
        self.speaker = ray.get(aiaa.tts_actr.get_rand_voice_id.remote())
        self.speaker = 6852
        self.llm_text_cb = partial(aiaa.aia_actr.text_out.remote, sess_id=self.id)
        self.llm_session_textin = partial(aiaa.llm_actr.llm_session_textin.remote, sess_id=self.llm_sess_id)
        self.llm_session_context_add = partial(aiaa.llm_actr.llm_session_context_add.remote,
                                               sess_id=self.llm_sess_id)
        si = new_sess.sess_info
        self.n2w = NumbersToWords()
        self.text_in_buffer = []
        self.text_to_llm(f'<Incoming call from "{si.from_name}" at "{si.from_number}">')
        print(f'Agent {self.speaker} at your service.')

    def text_to_llm(self, text:str):
        req = LLMRequest(text, self.llm_text_cb)
        req.auto_ctx_add = False
        self.llm_session_textin(req=req)
        self.last_llm_req_id = req.id

    def text_in(self, result:Union[STTResult,STTSentinel]):
        if isinstance(result, STTResult):
            if self.debug:
                print(f'STT: "{result.text=}" {result.no_speech_prob=}')
            nsp = result.no_speech_prob
            if nsp > STTRequest.max_ns_prob or len(result.text) == 0:
                if result.duration < 5.0:
                    return
                text = f'<unaudible duration={result.duration} no_speech_probability={nsp}>'
            else:
                text = result.text
            self.text_in_buffer.append(text)
            if len(self.say_buffer) > 0:
                self.say_buffer = self.say_buffer[:1]
                if self.saying is not None:
                    self.llm_session_context_add(content='<sentence interrupted>', role='user')
                    self.tts_sess.stop_saying(self.saying)
                    self.saying = None
            return
        if len(self.text_in_buffer) == 0:
            return
        text = ' '.join(self.text_in_buffer)
        self.text_in_buffer = []
        self.text_to_llm(text)
        return

    def text_out(self, result:LLMResult):
        if self.debug: print(f'text_out({result.text=})')
        if result.req_id != self.last_llm_req_id:
            print(f'LLMResult for old req_id: {result.req_id}')
            return
        if result.text == '<nothingtosay>':
            print(f'LLMResult: nothing to say')
            return
        text = sent_tokenize(result.text)
        out_sents = [text.pop(0),]
        for t in text:
            if len(out_sents[-1]) + len(t) < 128 or out_sents[-1].endswith(' i.e.'):
                out_sents[-1] += ' ' + t
            else:
                out_sents.append(t)
        for t in out_sents:
            self.tts_say(t)

    def _tts_say(self, tr:TTSRequest):
        self.saying = self.tts_sess.say(tr)
        self.llm_session_context_add(content=tr.text[0], role='assistant')

    def tts_say(self, text):
        if self.debug: print(f'tts_say({text=})')
        text = self.n2w(text)
        tts_req = TTSRequest([text,], done_cb=self.tts_say_done_cb, speaker_id=self.speaker)
        self.say_buffer.append(tts_req)
        if len(self.say_buffer) > 1:
            return
        self._tts_say(tts_req)

    def tts_say_done(self):
        if self.debug: print(f'tts_say_done()')
        tbuf = self.say_buffer
        tbuf.pop(0)
        if len(tbuf) > 0:
            self._tts_say(tbuf[0])
            return
        self.saying = None

    def sess_term(self, _):
        self.stt_sess_term()
        self.tts_sess.end()
        self.llm_sess_term()
