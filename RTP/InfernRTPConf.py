from typing import Optional

from sippy.Network_server import RTP_port_allocator

class InfernRTPConf():
    schema: dict = {
        'settings': {
            'type': 'dict',
            'schema': {
                'min_port': {'type': 'integer', 'min': 1, 'max': 65535},
                'max_port': {'type': 'integer', 'min': 1, 'max': 65535},
            }
        }
    }
    palloc: RTP_port_allocator
    def __init__(self, conf:Optional[dict]=None):
        max_port = conf.get('max_port', None) if conf is not None else None
        min_port = conf.get('min_port', None) if conf is not None else None
        self.palloc = RTP_port_allocator(min_port, max_port)
