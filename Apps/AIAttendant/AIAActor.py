from typing import Dict, Optional, List, Union
from uuid import UUID
from functools import partial

from ray import ray
import nltk
from tensorboardX import SummaryWriter

from config.InfernGlobals import InfernGlobals as IG
from Cluster.InfernSIPActor import InfernSIPActor
from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernSTTActor import InfernSTTActor
from Cluster.InfernLLMActor import InfernLLMActor
from Cluster.STTSession import STTResult, STTSentinel
from Cluster.LLMSession import LLMResult
from SIP.RemoteSession import RemoteSessionOffer
from Core.T2T.NumbersToWords import NumbersToWords
from Core.Exceptions.InfernSessNotFoundErr import InfernSessNotFoundErr

from .AIASession import AIASession
from ..LiveTranslator.LTActor import ntw_filter

class AIASessNotFoundErr(InfernSessNotFoundErr): pass

@ray.remote(resources={"ai_attendant": 1})
class AIAActor():
    sessions: Dict[UUID, AIASession]
    thunmbstones: List[UUID]
    translator: callable
    nstts: int = 0
    def __init__(self):
        self.stt_out_lang = 'en'

    def start(self, aia_prof: 'AIAProfile', sip_actr:InfernSIPActor):
        self.aia_prof = aia_prof
        self.tts_lang = aia_prof.tts_lang
        self.stt_lang = aia_prof.stt_lang
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.aia_actr = ray.get_runtime_context().current_actor
        self.sip_actr = sip_actr
        self.tts_actr = InfernTTSActor.remote()
        self.stt_actr = InfernSTTActor.remote()
        self.llm_actr = InfernLLMActor.remote()
        futs = [self.stt_actr.start.remote(),  self.tts_actr.start.remote(lang=self.tts_lang, output_sr=8000),
                self.llm_actr.start.remote()]
        if self.stt_out_lang == self.tts_lang:
            self.translator = ntw_filter
        else:
            flt = partial(ntw_filter, obj=NumbersToWords(self.tts_lang))
            self.translator = IG.get_translator(self.stt_out_lang, self.tts_lang, filter=flt).translate
        self.swriter = SummaryWriter()
        ray.get(futs)
        self.sessions = {}
        self.thumbstones = []

    def new_sip_session_received(self, new_sess:RemoteSessionOffer):
        aia_sess = AIASession(self, new_sess)
        print(f'{aia_sess=}')
        self.sessions[aia_sess.id] = aia_sess

    def sess_term(self, sess_id:UUID, sip_sess_id:UUID, relaxed:bool=False):
        try:
            self._get_session(sess_id).sess_term(sip_sess_id)
        except AIASessNotFoundErr:
            if not relaxed: raise
            return
        del self.sessions[sess_id]
        self.thumbstones.append(sess_id)
        if len(self.thumbstones) > 100:
            self.thumbstones = self.thumbstones[-100:]

    def text_in(self, sess_id:UUID, result:Union[STTResult,STTSentinel]):
        if isinstance(result, STTResult):
            self.swriter.add_scalar(f'stt/inf_time', result.inf_time, self.nstts)
            self.nstts += 1
        self._get_session(sess_id).text_in(result)

    def text_out(self, sess_id:UUID, result:LLMResult):
        try:
            self._get_session(sess_id).text_out(result)
        except AIASessNotFoundErr:
            if not sess_id in self.thumbstones: raise

    def tts_say_done(self, sess_id:UUID):
        self._get_session(sess_id).tts_say_done()

    def _get_session(self, sess_id:UUID):
        try: return self.sessions[sess_id]
        except KeyError: raise AIASessNotFoundErr(f'No AIA session with id {sess_id}')
