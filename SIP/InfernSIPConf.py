from typing import Optional
from os.path import expanduser

from sippy.SipConf import SipConf
from sippy.SipLogger import SipLogger

from Core.ConfigValidators import validate_port_range

class InfernSIPConf():
    schema: dict = {
        'settings': {
            'type': 'dict',
            'schema': {
                'bind': {
                    'type': 'string',
                    'regex': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:[1-9][0-9]{0,4}|:0)?$',
                    'check_with': validate_port_range
                }
            }
        }
    }
    logger = None

    def __init__(self, conf:Optional[dict]=None):
        self.logger = SipLogger('Infernos',  logfile = expanduser('~/.Infernos.log'))
        if conf is not None:
            try:
                bind = conf['bind'].split(':', 1)
            except KeyError: pass
            else:
                port = int(bind[1]) if len(bind) == 2 else SipConf.my_port
                self.laddr = bind[0]
                self.lport = port
                return
        self.laddr = SipConf.my_address
        self.lport = SipConf.my_port
