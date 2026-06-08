import importlib


def test_role_specific_openai_keys_fall_back_to_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "default-key")
    monkeypatch.setenv("OPENAI_PROSECUTOR_API_KEY", "prosecutor-key")
    monkeypatch.setenv("OPENAI_WITNESS_API_KEY", "witness-key")
    monkeypatch.setenv("OPENAI_JUDGE_API_KEY", "judge-key")
    monkeypatch.setenv("OPENAI_SYSTEM_API_KEY", "system-key")

    import backend.config as config

    config = importlib.reload(config)

    assert config.get_openai_api_key("prosecutor") == "prosecutor-key"
    assert config.get_openai_api_key("witness") == "witness-key"
    assert config.get_openai_api_key("judge") == "judge-key"
    assert config.get_openai_api_key("system") == "system-key"
    assert config.get_openai_api_key("unknown") == "default-key"


def test_explicit_api_key_override_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "default-key")
    monkeypatch.setenv("OPENAI_PROSECUTOR_API_KEY", "prosecutor-key")

    import backend.config as config

    config = importlib.reload(config)

    assert config.get_openai_api_key("prosecutor", "override-key") == "override-key"
