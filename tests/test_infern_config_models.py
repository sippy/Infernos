import importlib
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Core.ConfigValidators import InfernConfigParseErr


@contextmanager
def stub_infernconfig_dependencies():
    """Stub only the heavyweight modules required to import ``InfernConfig``."""

    def make_package(name: str) -> ModuleType:
        module = ModuleType(name)
        module.__path__ = []  # Mark as package so submodules may be imported
        return module

    saved_modules: Dict[str, Optional[ModuleType]] = {}

    def stub_module(name: str, module: ModuleType) -> None:
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    # Provide lightweight stand-ins for the external ``sippy`` dependency used by
    # the SIP/RTP validators that ``InfernConfig`` imports during module load.
    stub_module("sippy", make_package("sippy"))

    sipconf_module = ModuleType("sippy.SipConf")

    class _SipConf:
        my_port = 5060
        my_address = "127.0.0.1"

    sipconf_module.SipConf = _SipConf
    stub_module("sippy.SipConf", sipconf_module)

    siplogger_module = ModuleType("sippy.SipLogger")

    class _SipLogger:
        def __init__(self, *_args, **_kwargs):
            pass

    siplogger_module.SipLogger = _SipLogger
    stub_module("sippy.SipLogger", siplogger_module)

    network_module = ModuleType("sippy.Network_server")

    class _RTPAllocator:
        def __init__(self, min_port=None, max_port=None):
            self.min_port = min_port
            self.max_port = max_port

    network_module.RTP_port_allocator = _RTPAllocator
    stub_module("sippy.Network_server", network_module)

    # Stub Cluster actor dependency to avoid importing ray.
    stub_module("Cluster", make_package("Cluster"))
    cluster_module = ModuleType("Cluster.InfernSIPActor")

    class DummyActor:
        @classmethod
        def options(cls, **_kwargs):
            class _Opts:
                def remote(self_inner):  # noqa: N805 - mimic Ray API
                    return cls()

            return _Opts()

    cluster_module.InfernSIPActor = DummyActor
    stub_module("Cluster.InfernSIPActor", cluster_module)

    # Stub app profiles that pull in ray and other runtime dependencies.
    stub_module("Apps", make_package("Apps"))

    live_translator_pkg = make_package("Apps.LiveTranslator")
    stub_module("Apps.LiveTranslator", live_translator_pkg)
    lt_profile_module = ModuleType("Apps.LiveTranslator.LTProfile")

    class DummyLTProfile:
        schema = {
            "profiles": {
                "type": "dict",
                "keysrules": {"type": "string"},
                "valuesrules": {"type": "dict", "schema": {}},
            }
        }

        def __init__(self, name, conf, precache):
            self.name = name
            self.conf = conf
            self.precache = precache

        def finalize(self, *_args, **_kwargs):
            return None

        def getActor(self, *_args, **_kwargs):  # noqa: N802 - mimic original API
            return None

    lt_profile_module.LTProfile = DummyLTProfile
    stub_module("Apps.LiveTranslator.LTProfile", lt_profile_module)

    lt_app_config_module = ModuleType("Apps.LiveTranslator.LTAppConfig")

    class DummyLTAppConfig:
        schema = {
            "live_translator": {
                "type": "dict",
                "schema": DummyLTProfile.schema,
            },
            "live_translator_precache": {"type": "boolean"},
        }

    lt_app_config_module.LTAppConfig = DummyLTAppConfig
    stub_module("Apps.LiveTranslator.LTAppConfig", lt_app_config_module)

    ai_attendant_pkg = make_package("Apps.AIAttendant")
    stub_module("Apps.AIAttendant", ai_attendant_pkg)
    aia_profile_module = ModuleType("Apps.AIAttendant.AIAProfile")

    class DummyAIAProfile:
        schema = {
            "profiles": {
                "type": "dict",
                "keysrules": {"type": "string"},
                "valuesrules": {"type": "dict", "schema": {}},
            }
        }

        def __init__(self, name, conf):
            self.name = name
            self.conf = conf

        def finalize(self, *_args, **_kwargs):
            return None

        def getActor(self, *_args, **_kwargs):  # noqa: N802 - mimic original API
            return None

    aia_profile_module.AIAProfile = DummyAIAProfile
    stub_module("Apps.AIAttendant.AIAProfile", aia_profile_module)

    aia_app_config_module = ModuleType("Apps.AIAttendant.AIAAppConfig")

    class DummyAIAAppConfig:
        schema = {
            "ai_attendant": {
                "type": "dict",
                "schema": DummyAIAProfile.schema,
            }
        }

    aia_app_config_module.AIAAppConfig = DummyAIAAppConfig
    stub_module("Apps.AIAttendant.AIAAppConfig", aia_app_config_module)

    # Finally stub the STT modules so importing model profiles is lightweight.
    if "Models.STT" not in sys.modules:
        stub_module("Models.STT", make_package("Models.STT"))
    whisper_module = ModuleType("Models.STT.Whisper")

    class FakeWhisper:
        provides = "whisper"
        schema = {
            "device": {"type": "string", "allowed": ["cpu", "cuda"]},
            "model_name": {"type": "string"},
        }

    whisper_module.Whisper = FakeWhisper
    stub_module("Models.STT.Whisper", whisper_module)

    whisper_rt_module = ModuleType("Models.STT.WhisperRT")

    class FakeWhisperRT:
        provides = "whisper_rt"
        schema = {
            "device": {"type": "string", "allowed": ["auto", "cpu", "cuda"]},
            "beam_size": {"type": "integer", "min": 1, "coerce": int},
        }

    whisper_rt_module.WhisperRT = FakeWhisperRT
    stub_module("Models.STT.WhisperRT", whisper_rt_module)

    try:
        yield
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(name="InfernConfig")
def infern_config_fixture():
    with stub_infernconfig_dependencies():
        for module_name in (
            "Core.InfernConfig",
            "Models.InfernModelConf",
            "Models.InfernModelProfile",
        ):
            sys.modules.pop(module_name, None)
        module = importlib.import_module("Core.InfernConfig")
        yield module.InfernConfig


def test_infern_config_parses_model_profiles(tmp_path, InfernConfig):
    config_path = tmp_path / "models_valid.yaml"
    config_path.write_text(
        """
models:
  settings:
    cache_dir: ~/models
  profiles:
    default:
      uses: whisper
      parameters:
        device: cuda
        model_name: openai/whisper-test
    fast:
      uses: whisper_rt
      batch_size: 8
      parameters:
        device: auto
        beam_size: "2"
apps: {}
""".strip()
    )

    config = InfernConfig(str(config_path))

    assert config.model_conf.cache_dir.endswith("/models")
    assert set(config.model_conf.profiles) == {"default", "fast"}
    assert config.model_conf.profiles["default"].parameters == {
        "device": "cuda",
        "model_name": "openai/whisper-test",
    }
    assert config.model_conf.profiles["fast"].parameters == {
        "device": "auto",
        "beam_size": 2,
    }
    assert config.model_conf.profiles["fast"].batch_size == 8


def test_invalid_model_parameters_raise(tmp_path, InfernConfig):
    config_path = tmp_path / "models_invalid.yaml"
    config_path.write_text(
        """
models:
  profiles:
    invalid:
      uses: whisper
      parameters:
        device: tpu
apps: {}
""".strip()
    )

    with pytest.raises(InfernConfigParseErr):
        InfernConfig(str(config_path))


def test_unknown_model_raises(tmp_path, InfernConfig):
    config_path = tmp_path / "models_unknown.yaml"
    config_path.write_text(
        """
models:
  profiles:
    bad:
      uses: made_up_model
apps: {}
""".strip()
    )

    with pytest.raises(InfernConfigParseErr):
        InfernConfig(str(config_path))
