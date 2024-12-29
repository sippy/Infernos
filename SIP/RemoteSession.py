from typing import Optional, Tuple
from functools import partial
from uuid import UUID

from SIP.SipSessInfo import SipSessInfo
from .InfernSIPProfile import InfernSIPProfile

class RemoteSessionOffer():
    sip_sess_id: UUID
    sess_info: SipSessInfo
    accept: callable
    reject: callable
    def __init__(self, sip_stack:'InfernSIP', ua:'InfernLazyUAS'):
        self.sip_sess_id = ua.id
        self.sess_info = ua.get_session_info()
        self.accept = partial(sip_stack.sip_actr.new_sess_accept.remote, sip_sess_id=ua.id)
        self.reject = partial(sip_stack.sip_actr.new_sess_reject.remote, sip_sess_id=ua.id)

class RemoteSessionAccept():
    disc_cb: Optional[callable] = None
    auto_answer: bool = False
    def __init__(self, disc_cb:Optional[callable]=None, auto_answer:bool=False):
        self.disc_cb, self.auto_answer = disc_cb, auto_answer

class NewRemoteSessionRequest():
    cld:str
    sip_prof: InfernSIPProfile
    disc_cb: Optional[callable] = None
    conn_sip_sess_id: Optional[UUID] = None
    def __init__(self, cld:str, sip_prof:InfernSIPProfile, disc_cb:Optional[callable]=None):
        self.cld, self.disc_cb, self.sip_prof = cld, disc_cb, sip_prof
