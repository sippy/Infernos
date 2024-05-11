from typing import Optional, Tuple
from functools import partial

from Cluster.InfernSIPActor import InfernSIPActor

class InfernSIPProfile():
    name: str
    cli: str = 'infernos_uas'
    authname: Optional[str] = None
    authpass: Optional[str] = None
    nh_addr: Optional[Tuple[str, int]] = None
    register: bool = False
    _sink: str
    new_sess_offer: callable = None

    def __init__(self, name, conf):
        self.name = name
        self.cli = conf.get('username', self.cli)
        self.authname = conf.get('username', self.authname)
        self.authpass = conf.get('password', self.authpass)
        sip_server = conf['sip_server'].split(':', 1)
        port = int(sip_server[1]) if len(sip_server) == 2 else 5060
        self.nh_addr = (sip_server[0], port)
        self.register = conf.get('register', self.register)
        self._sink = conf['sink']

    def finalize(self, sip_actr: InfernSIPActor, iconf: 'InfernConfig'):
        sact = iconf.apps[self._sink].getActor(iconf, sip_actr)
        self.new_sess_offer = partial(sact.new_sip_session_received.remote)
