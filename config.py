import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EPISODES_DIR = DATA_DIR / "episodes"

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_PROSECUTOR_API_KEY = _first_env(
    "OPENAI_PROSECUTOR_API_KEY",
    "OPENAI_API_KEY_PROSECUTOR",
    "PROSECUTOR_OPENAI_API_KEY",
)
OPENAI_WITNESS_API_KEY = _first_env(
    "OPENAI_WITNESS_API_KEY",
    "OPENAI_API_KEY_WITNESS",
    "WITNESS_OPENAI_API_KEY",
)
OPENAI_JUDGE_API_KEY = _first_env(
    "OPENAI_JUDGE_API_KEY",
    "OPENAI_API_KEY_JUDGE",
    "JUDGE_OPENAI_API_KEY",
)
OPENAI_SYSTEM_API_KEY = _first_env(
    "OPENAI_SYSTEM_API_KEY",
    "OPENAI_API_KEY_SYSTEM",
    "SYSTEM_OPENAI_API_KEY",
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_ACTOR_MODEL = os.getenv("OPENAI_ACTOR_MODEL", "gpt-4.1")
OPENAI_PROSECUTOR_MODEL = os.getenv("OPENAI_PROSECUTOR_MODEL", OPENAI_ACTOR_MODEL)
OPENAI_WITNESS_MODEL = os.getenv("OPENAI_WITNESS_MODEL", OPENAI_ACTOR_MODEL)
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", OPENAI_ACTOR_MODEL)
OPENAI_VERIFIER_MODEL = os.getenv("OPENAI_VERIFIER_MODEL", OPENAI_MODEL)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
USE_REDIS = os.getenv("USE_REDIS", "false").lower() in ("1", "true", "yes")
DATABASE_PATH = os.getenv("DATABASE_PATH", str(DATA_DIR / "game.db"))

OPENAI_ROLE_API_KEYS = {
    "prosecutor": OPENAI_PROSECUTOR_API_KEY,
    "witness": OPENAI_WITNESS_API_KEY,
    "judge": OPENAI_JUDGE_API_KEY,
    "system": OPENAI_SYSTEM_API_KEY,
}


def get_openai_api_key(role: str | None = None, override: str | None = None) -> str:
    if override:
        return override
    if role:
        role_key = OPENAI_ROLE_API_KEYS.get(role, "")
        if role_key:
            return role_key
    return OPENAI_API_KEY
