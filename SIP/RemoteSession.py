from typing import Optional
from functools import partial
from uuid import UUID

class RemoteSessionOffer():
    sip_sess_id: UUID
    accept: callable
    reject: callable
    def __init__(self, sip_stack:'InfernSIP', ua:'InfernLazyUAS'):
        self.sip_sess_id = ua.id
        self.accept = partial(sip_stack.sip_actr.new_sess_accept.remote, sip_sess_id=ua.id)
        self.reject = partial(sip_stack.sip_actr.new_sess_reject.remote, sip_sess_id=ua.id)

class RemoteSessionAccept():
    disc_cb: Optional[callable] = None
    auto_answer: bool = False
    def __init__(self, disc_cb:Optional[callable]=None, auto_answer:bool=False):
        self.disc_cb, self.auto_answer = disc_cb, auto_answer

class NewRemoteSessionRequest():
    cld:str
    disc_cb: Optional[callable] = None
    conn_sip_sess_id: Optional[UUID] = None
    def __init__(self, cld:str, disc_cb:Optional[callable]=None):
        self.cld, self.disc_cb = cld, disc_cb
