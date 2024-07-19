from typing import Optional, Tuple
from functools import partial

from Core.ConfigValidators import validate_port_range

class InfernSIPProfile():
    schema: dict = {
        'profiles': {
            'type': 'dict',
            'keysrules': {'type': 'string'},
            'valuesrules': {
                'type': 'dict',
                'schema': {
                    'sip_server': {
                        'type': 'string',
                        'regex': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:[1-9][0-9]{0,4}|:0)?$',
                        'check_with': validate_port_range
                    },
                    'aor': {'type': 'string'},
                    'username': {'type': 'string'},
                    'password': {'type': 'string'},
                    'register': {'type': 'boolean'},
                    'sink': {'type': 'string'},
                }
            }
        }
    }
    name: str
    cli: str = 'infernos_uas'
    aor: str
    authname: Optional[str] = None
    authpass: Optional[str] = None
    nh_addr: Optional[Tuple[str, int]] = None
    register: bool = False
    _sink: Optional[str]
    new_sess_offer: callable = None

    def __init__(self, name, conf):
        self.name = name
        self.cli = conf.get('username', self.cli)
        self.aor = conf.get('aor', self.cli)
        self.authname = conf.get('username', self.authname)
        self.authpass = conf.get('password', self.authpass)
        sip_server = conf['sip_server'].split(':', 1)
        port = int(sip_server[1]) if len(sip_server) == 2 else 5060
        self.nh_addr = (sip_server[0], port)
        self.register = conf.get('register', self.register)
        self._sink = conf.get('sink', None)

    def finalize(self, sip_actr: 'InfernSIPActor', iconf: 'InfernConfig'):
        if self._sink is None: return
        sact = iconf.apps[self._sink].getActor(iconf, sip_actr)
        self.new_sess_offer = partial(sact.new_sip_session_received.remote)
