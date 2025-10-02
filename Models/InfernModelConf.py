from typing import Dict, Optional
from os.path import expanduser

from .InfernModelProfile import InfernModelProfile

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
                'schema': InfernModelProfile.schema,
            }
        }
    }
    cache_dir: str = '~/.cache/Infernos'
    profiles: Dict[str, InfernModelProfile]

    def __init__(self, conf: Optional[dict] = None):
        conf = conf or {}
        settings = conf.get('settings') or {}
        cdir = settings.get('cache_dir', self.cache_dir)
        self.cache_dir = expanduser(cdir)
        profiles_conf = conf.get('profiles') or {}
        self.profiles = {
            name: InfernModelProfile(name, profile_conf)
            for name, profile_conf in profiles_conf.items()
        }
