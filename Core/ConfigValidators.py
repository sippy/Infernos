import yaml
from cerberus import Validator

class InfernConfigParseErr(Exception): pass

def validate_yaml(schema, filename):
    try:
        with open(filename, 'r') as file:
            data = yaml.safe_load(file)

        v = Validator(schema)
        if not v.validate(data):
            raise InfernConfigParseErr(f"Validation errors in {filename}: {v.errors}")

    except yaml.YAMLError as exc:
        raise InfernConfigParseErr(f"Error parsing YAML file {filename}: {exc}") from exc
    return data

def validate_port_range(field, value, error):
    if ':' in value:
        _, port = value.split(':', 1)
        if not (1 <= int(port) <= 65535):
            error(field, 'Port number must be in the range 1-65535')
