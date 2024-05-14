from typing import Dict, Optional, List
from uuid import UUID
from functools import partial

from ray import ray
import nltk
from tensorboardX import SummaryWriter

from config.InfernGlobals import InfernGlobals as IG
from Cluster.InfernSIPActor import InfernSIPActor
from Cluster.InfernTTSActor import InfernTTSActor
from Cluster.InfernSTTActor import InfernSTTActor
from Cluster.STTSession import STTResult
from SIP.RemoteSession import RemoteSessionOffer
from Core.T2T.NumbersToWords import NumbersToWords
from Core.Exceptions.InfernSessNotFoundErr import InfernSessNotFoundErr

from .LTSession import LTSession, VADSignals

def ntw_filter(text, from_code=None, to_code=None, tr=lambda x:x, obj=NumbersToWords()):
    print(f'ntw_filter({from_code=}, {to_code=}, {text=})')
    return obj(tr(text))

class LTSessNotFoundErr(InfernSessNotFoundErr): pass

@ray.remote(resources={"live_translator": 1})
class LTActor():
    sessions: Dict[UUID, LTSession]
    vds: Optional[VADSignals]=None
    translators: List[callable]
    nstts: int = 0
    def __init__(self):
        self.tts_langs = ('it', 'en')
        self.stt_langs = ('en', 'it')
        self.stt_out_langs = ('en', 'en')

    def start(self, sip_actr:InfernSIPActor):
        nltk.download('punkt')
        self.lt_actr = ray.get_runtime_context().current_actor
        self.sip_actr = sip_actr
        self.tts_actrs = dict((l, InfernTTSActor.remote()) for l in self.tts_langs)
        self.stt_actr = InfernSTTActor.remote()
        futs = [_a.start.remote(**_k) for _a, _k in ((self.stt_actr, {}),) +
                         tuple((a, {'lang':l, 'output_sr':8000}) for l, a in self.tts_actrs.items())]
        self.translators = [ntw_filter if _sol == _tl else
                            IG.get_translator(_sol, _tl, filter=partial(ntw_filter, obj=NumbersToWords(_tl))).translate
                            for _tl, _sol in zip(self.tts_langs, self.stt_out_langs)]
        self.swriter = SummaryWriter()
        ray.get(futs)
        self.sessions = {}

    def new_sip_session_received(self, new_sess:RemoteSessionOffer):
        if self.vds is None:
            self.vds = VADSignals()
        lt_sess = LTSession(self, new_sess)
        print(f'{lt_sess=}')
        self.sessions[lt_sess.id] = lt_sess

    def sess_term(self, sess_id:UUID, sip_sess_id:UUID, relaxed:bool=False):
        try:
            self._get_session(sess_id).sess_term(sip_sess_id)
        except LTSessNotFoundErr:
            if not relaxed: raise
            return
        del self.sessions[sess_id]

    def text_in(self, sess_id:UUID, result:STTResult):
        self.swriter.add_scalar(f'stt/inf_time', result.inf_time, self.nstts)
        self.nstts += 1
        self._get_session(sess_id).text_in(result)

    def tts_say_done(self, sess_id:UUID, direction:int):
        self._get_session(sess_id).tts_say_done(direction)

    def _get_session(self, sess_id:UUID):
        try: return self.sessions[sess_id]
        except KeyError: raise LTSessNotFoundErr(f'No LT session with id {sess_id}')
