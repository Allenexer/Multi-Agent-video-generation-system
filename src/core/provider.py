"""
Provider abstraction — encapsulates an LLM API endpoint.
Decouples agent logic from specific API providers.
"""
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class ModelInfo:
    """Metadata about an available model."""
    id: str            # API model ID, e.g. "deepseek-ai/DeepSeek-V3.2"
    name: str          # Display name, e.g. "DeepSeek-V3.2"
    type: str          # "text" | "vision"
    description: str = ""


@dataclass
class Provider:
    """
    An LLM API provider. Encapsulates base_url + api_key.
    Creates OpenAI-compatible clients on demand.
    """
    id: str
    name: str
    base_url: str
    api_key: str
    models: list[ModelInfo] = field(default_factory=list)

    def create_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_model_ids(self, model_type: str = None) -> list[str]:
        """Return model IDs, optionally filtered by type."""
        if model_type:
            return [m.id for m in self.models if m.type == model_type]
        return [m.id for m in self.models]

    def get_default_model(self, model_type: str = "text") -> str:
        """Return the first model of given type, or the first model overall."""
        matching = self.get_model_ids(model_type)
        if matching:
            return matching[0]
        if self.models:
            return self.models[0].id
        raise ValueError(f"No models configured for provider '{self.id}'")

    def find_free_model(self, model_type: str = "text") -> str | None:
        """Pick a low-cost model of given type if available.
        Heuristic: model id contains 'flash'/'free'/'mini'/'lite',
        or description contains '免费'/'free'.
        """
        keywords_id = ("flash", "free", "mini", "lite")
        keywords_desc = ("免费", "free")
        for m in self.models:
            if model_type and m.type != model_type:
                continue
            dl = (m.description or "").lower()
            il = m.id.lower()
            if any(k in dl for k in keywords_desc) or \
               any(k in il for k in keywords_id):
                return m.id
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "models": [
                {"id": m.id, "name": m.name, "type": m.type, "description": m.description}
                for m in self.models
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Provider":
        models = [
            ModelInfo(
                id=m["id"], name=m["name"], type=m["type"],
                description=m.get("description", ""),
            )
            for m in data.get("models", [])
        ]
        return cls(
            id=data["id"], name=data["name"],
            base_url=data["base_url"], api_key=data["api_key"],
            models=models,
        )
