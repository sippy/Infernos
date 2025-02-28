from typing import Dict, Optional, Union
from functools import partial

from Cluster.InfernSIPActor import InfernSIPActor
from SIP.InfernSIPConf import InfernSIPConf
from SIP.InfernSIPProfile import InfernSIPProfile
from RTP.InfernRTPConf import InfernRTPConf

from .ConfigValidators import validate_yaml

# Define the schema
schema = {
    'sip': {
        'type': 'dict',
        'schema': {
            **InfernSIPConf.schema,
            **InfernSIPProfile.schema,
        }
    },
    'rtp': {
        'type': 'dict',
        'schema': {
            **InfernRTPConf.schema,
        }
    },
    'apps': {
        'type': 'dict',
        'schema': {
            # Filled by modules
        }
    }
}

class InfernConfig():
    sip_actr: Optional[InfernSIPActor]
    sip_conf: Optional[InfernSIPConf]
    rtp_conf: Optional[InfernRTPConf]
    connectors: Dict[str, InfernSIPProfile]
    apps: Dict[str, Union['LTProfile', 'AIAProfile']]
    def __init__(self, filename: str):
        from Apps.LiveTranslator.LTProfile import LTProfile
        from Apps.LiveTranslator.LTAppConfig import LTAppConfig
        schema['apps']['schema'].update(LTAppConfig.schema)
        from Apps.AIAttendant.AIAProfile import AIAProfile
        from Apps.AIAttendant.AIAAppConfig import AIAAppConfig
        schema['apps']['schema'].update(AIAAppConfig.schema)
        d = validate_yaml(schema, filename)
        self.sip_conf = InfernSIPConf(d['sip'].get('settings', None)) if 'sip' in d else None
        self.rtp_conf = InfernRTPConf(d['rtp'].get('settings', None)) if 'rtp' in d else None
        try:
            self.connectors = dict((f'sip/{name}', InfernSIPProfile(name, conf))
                                for name, conf in d['sip']['profiles'].items())
        except KeyError:
            self.connectors = {}
        precache = 'live_translator_precache' in d['apps'] and d['apps']['live_translator_precache']
        _LTProfile = partial(LTProfile, precache=precache)
        self.apps = {}
        for aname, AProf in (('live_translator', _LTProfile), ('ai_attendant', AIAProfile)):
            if aname not in d['apps']: continue
            app_confs = dict((f'apps/{aname}/{name}', AProf(name, conf))
                        for name, conf in d['apps'][aname]['profiles'].items())
            self.apps.update(app_confs)
        for app in self.apps.values():
            app.finalize(self)
        if 'sip' in d:
            self.sip_actr = InfernSIPActor.options(max_concurrency=2).remote()
            for conn in self.connectors.values():
                conn.finalize(self.sip_actr, self)
        else:
            self.sip_actr = None
