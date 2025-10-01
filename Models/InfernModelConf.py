from typing import Optional
from os.path import expanduser

class InfernModelConf():
    schema: dict = {
        'settings': {
            'type': 'dict',
            'schema': {
                'cache_dir': { 'type': 'string' }
            }
        },
        'profiles': {
            'type': 'dict',
            'keysrules': {'type': 'string'},
            'valuesrules': {
                'type': 'dict',
                'schema': {
                    'uses': {'type': 'string'},
                }
            }
        }
    }
    cache_dir: str = '~/.cache/Infernos'

    def __init__(self, conf:Optional[dict]=None):
        cdir = conf['cache_dir'] if conf is not None and 'cache_dir' in conf else self.cache_dir
        self.cache_dir = expanduser(cdir)
