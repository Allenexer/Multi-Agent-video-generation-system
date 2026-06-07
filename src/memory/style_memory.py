"""
StyleMemory — persists style analysis results across sessions.

When the same reference image or style description is reused,
retrieve cached analysis instead of re-calling VLM.
"""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parent / "data"
STORE_FILE = DATA_DIR / "styles.json"


class StyleMemory:
    """
    JSON-backed style analysis cache.

    Keyed by MD5 hash of either:
      - Image file path + file modification time (for reference images)
      - Style description text (for text-described styles)
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

    def get_by_image(self, image_path: str) -> dict | None:
        """Retrieve cached style analysis for a reference image."""
        import os
        if not os.path.exists(image_path):
            return None
        mtime = os.path.getmtime(image_path)
        key = hashlib.md5(f"{image_path}|{mtime}".encode()).hexdigest()[:12]
        entry = self._store.get(key)
        if entry is None:
            return None
        created = datetime.fromisoformat(entry.get("created_at", "2000-01-01"))
        if datetime.now() - created > timedelta(days=30):
            return None
        return entry["analysis"]

    def put_by_image(self, image_path: str, analysis: dict):
        """Cache style analysis for a reference image."""
        import os
        mtime = os.path.getmtime(image_path)
        key = hashlib.md5(f"{image_path}|{mtime}".encode()).hexdigest()[:12]
        self._store[key] = {
            "analysis": analysis,
            "created_at": datetime.now().isoformat(),
            "source": image_path,
        }
        self._save()

    def get_by_text(self, style_description: str) -> dict | None:
        """Retrieve cached style for a text description."""
        key = hashlib.md5(style_description.encode()).hexdigest()[:12]
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry["analysis"]

    def put_by_text(self, style_description: str, analysis: dict):
        """Cache style analysis for a text description."""
        key = hashlib.md5(style_description.encode()).hexdigest()[:12]
        self._store[key] = {
            "analysis": analysis,
            "created_at": datetime.now().isoformat(),
            "source": style_description[:80],
        }
        self._save()
