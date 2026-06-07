"""
CharacterMemory — persists character designs across sessions.

When the same character appears in multiple videos, reuse its design
instead of re-generating (saves API calls + ensures consistency).
"""
import json
import os
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent / "data"
STORE_FILE = DATA_DIR / "characters.json"


class CharacterMemory:
    """
    JSON-backed character design store.

    Each entry keyed by a hash of (character_name, key_traits).
    Stores: appearance prompt fragment, consistency keywords, creation date.
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._store = self._load()

    def _load(self) -> dict:
        if STORE_FILE.exists():
            return json.loads(STORE_FILE.read_text(encoding="utf-8"))
        return {}

    def _save(self):
        STORE_FILE.write_text(
            json.dumps(self._store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _key(name: str, traits: str = "") -> str:
        return hashlib.md5(f"{name}|{traits}".encode()).hexdigest()[:12]

    def get(self, name: str, traits: str = "") -> dict | None:
        """Retrieve a stored character design, or None."""
        key = self._key(name, traits)
        entry = self._store.get(key)
        if entry is None:
            return None
        # Check expiry (90 days)
        created = datetime.fromisoformat(entry.get("created_at", "2000-01-01"))
        if datetime.now() - created > timedelta(days=90):
            return None
        return entry["design"]

    def put(self, name: str, traits: str, design: dict):
        """Store a character design."""
        key = self._key(name, traits)
        self._store[key] = {
            "design": design,
            "created_at": datetime.now().isoformat(),
            "name": name,
        }
        self._save()

    def list_all(self) -> list[dict]:
        """Return all stored characters (for UI display)."""
        return [
            {"key": k, "name": v["name"], "created_at": v["created_at"]}
            for k, v in self._store.items()
        ]
