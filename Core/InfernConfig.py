from typing import Dict

import yaml
from cerberus import Validator

def validate_port_range(field, value, error):
    if ':' in value:
        _, port = value.split(':', 1)
        if not (1 <= int(port) <= 65535):
            error(field, 'Port number must be in the range 1-65535')

# Define the schema
schema = {
    'sip': {
        'type': 'dict',
        'schema': {
            'settings': {
                'type': 'dict',
                'schema': {
                    'bind': {
                        'type': 'string',
                        'regex': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:[1-9][0-9]{0,4}|:0)?$',
                        'check_with': validate_port_range
                    },
                }
            },
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
    },
    'apps': {
        'type': 'dict',
        'schema': {
            'live_translator': {
                'type': 'dict',
                'schema': {
                    'profiles': {
                        'type': 'dict',
                        'keysrules': {'type': 'string'},
                        'valuesrules': {
                            'type': 'dict',
                            'schema': {
                                'tts_langs': {'type': 'list', 'schema': {'type': 'string'}},
                                'stt_langs': {'type': 'list', 'schema': {'type': 'string'}},
                                'outbound': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            'live_translator_precache': {'type': 'boolean'},
        }
    }
}

class InfernConfigParseErr(Exception): pass

def validate_yaml(filename):
    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)

        v = Validator(schema)
        if not v.validate(data):
            raise InfernConfigParseErr(f"Validation errors in {filename}: {v.errors}")

    except yaml.YAMLError as exc:
        raise InfernConfigParseErr(f"Error parsing YAML file {filename}: {exc}") from exc
    return data

from Cluster.InfernSIPActor import InfernSIPActor
from SIP.InfernUA import InfernSIPConf
from SIP.InfernSIPProfile import InfernSIPProfile

class InfernConfig():
    sip_actr: InfernSIPActor
    sip_conf: InfernSIPConf
    connectors: Dict[str, InfernSIPProfile]
    apps: Dict[str, 'LTProfile']
    def __init__(self, filename: str):
        from Apps.LiveTranslator.LTProfile import LTProfile
        d = validate_yaml(filename)
        self.sip_conf = InfernSIPConf()
        try:
            bind = d['sip']['settings']['bind'].split(':', 1)
        except KeyError: pass
        else:
            port = int(bind[1]) if len(bind) == 2 else self.sip_conf.lport
            self.sip_conf.laddr = bind[0]
            self.sip_conf.lport = port
        try:
            self.connectors = dict((f'sip/{name}', InfernSIPProfile(name, conf))
                                for name, conf in d['sip']['profiles'].items())
        except KeyError:
            self.connectors = {}
        precache = 'live_translator_precache' in d['apps'] and d['apps']['live_translator_precache']
        self.apps = dict((f'apps/live_translator/{name}', LTProfile(name, conf, precache))
                         for name, conf in d['apps']['live_translator']['profiles'].items())
        for app in self.apps.values():
            app.finalize(self)
        if 'sip' in d:
            self.sip_actr = InfernSIPActor.options(max_concurrency=2).remote()
            for conn in self.connectors.values():
                conn.finalize(self.sip_actr, self)
        else:
            self.sip_actr = None
