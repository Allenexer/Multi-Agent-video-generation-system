"""
Configuration manager — loads and manages providers, agents, pipeline.
Central source of truth for all runtime configuration.
"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from core.provider import Provider

load_dotenv()

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _resolve_env(value: str) -> str:
    """Resolve ${ENV_VAR} references in config values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        var_name = value[2:-1]
        return os.getenv(var_name, value)
    return value


class ConfigManager:
    """
    Singleton that loads and manages configuration from YAML files.

    Three config files:
      - providers.yaml  : API providers and their models
      - agents.yaml     : Agent definitions (prompts, model type requirements)
      - pipeline.yaml   : Pipeline/workflow structure
    """

    _instance = None

    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or CONFIG_DIR
        self.providers: dict[str, Provider] = {}
        self.agent_configs: dict[str, dict] = {}
        self.pipeline_config: dict = {}
        # Runtime overrides (modified by UI at runtime)
        self.agent_model_overrides: dict[str, str] = {}  # agent_id → model_id
        self.agent_provider_overrides: dict[str, str] = {}  # agent_id → provider_id
        self._reload()

    @classmethod
    def get_instance(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Load ──

    def _reload(self):
        self._load_providers()
        self._load_agents()
        self._load_pipeline()

    def _load_providers(self):
        path = self.config_dir / "providers.yaml"

        if not path.exists():
            raise FileNotFoundError(f"providers.yaml not found in {self.config_dir}")
        
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in data.get("providers", []):
            # Resolve env vars in api_key
            entry["api_key"] = _resolve_env(entry.get("api_key", ""))
            entry["base_url"] = _resolve_env(entry.get("base_url", ""))
            provider = Provider.from_dict(entry)
            self.providers[provider.id] = provider

    def _load_agents(self):
        path = self.config_dir / "agents.yaml"

        if not path.exists():
            raise FileNotFoundError(f"agents.yaml not found in {self.config_dir}")
        
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.agent_configs = data.get("agents", {})

    def _load_pipeline(self):
        path = self.config_dir / "pipeline.yaml"

        if not path.exists():
            raise FileNotFoundError(f"pipeline.yaml not found in {self.config_dir}")
        
        self.pipeline_config = yaml.safe_load(path.read_text(encoding="utf-8"))

    # ── Save ──

    def save_providers(self):
        path = self.config_dir / "providers.yaml"
        data = {"providers": [p.to_dict() for p in self.providers.values()]}
        path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False),
                       encoding="utf-8")

    def save_pipeline(self):
        path = self.config_dir / "pipeline.yaml"
        path.write_text(yaml.dump(self.pipeline_config, allow_unicode=True, default_flow_style=False),
                       encoding="utf-8")

    # ── Query ──

    def get_provider_for_agent(self, agent_id: str) -> Provider:
        """Resolve which provider an agent should use."""
        provider_id = self.agent_provider_overrides.get(agent_id)
        if provider_id and provider_id in self.providers:
            return self.providers[provider_id]
        # Fallback: use agent config's default provider
        agent_cfg = self.agent_configs.get(agent_id, {})
        default_provider = agent_cfg.get("default_provider", "")
        if default_provider in self.providers:
            return self.providers[default_provider]
        # Last resort: first available provider with matching model type
        model_type = agent_cfg.get("model_type", "text")
        for provider in self.providers.values():
            if provider.get_model_ids(model_type):
                return provider
        raise ValueError(f"No provider found for agent '{agent_id}' (model_type={model_type})")

    def get_model_for_agent(self, agent_id: str) -> str:
        """Resolve which model an agent should use.
        Priority: user override → preferred model → free model fallback → default.
        """
        override = self.agent_model_overrides.get(agent_id)
        if override:
            return override
        provider = self.get_provider_for_agent(agent_id)
        agent_cfg = self.agent_configs.get(agent_id, {})
        model_type = agent_cfg.get("model_type", "text")

        preferred = agent_cfg.get("preferred_model")
        if preferred and preferred in provider.get_model_ids():
            return preferred

        # Adaptive: cost-sensitive agents prefer a free model when available
        if self._should_use_free_model(agent_id, agent_cfg):
            free = provider.find_free_model(model_type)
            if free:
                return free

        return provider.get_default_model(model_type)

    @staticmethod
    def _should_use_free_model(agent_id: str, agent_cfg: dict) -> bool:
        """Heuristic: critic-style or prefer_free_model agents downgrade."""
        if agent_cfg.get("prefer_free_model"):
            return True
        aid_lc = (agent_id or "").lower()
        return any(k in aid_lc for k in ("critic", "review", "judge"))

    def get_client_for_agent(self, agent_id: str):
        """Get an OpenAI client configured for the given agent."""
        provider = self.get_provider_for_agent(agent_id)
        return provider.create_client()

    def get_agent_ids(self) -> list[str]:
        return list(self.agent_configs.keys())

    def get_agent_config(self, agent_id: str) -> dict:
        return self.agent_configs.get(agent_id, {})

    def get_pipeline_stages(self) -> list[dict]:
        return self.pipeline_config.get("stages", [])

    def get_pipeline_name(self) -> str:
        return self.pipeline_config.get("name", "默认流水线")

    # ── UI mutations ──

    def set_agent_model(self, agent_id: str, model_id: str):
        self.agent_model_overrides[agent_id] = model_id

    def set_agent_provider(self, agent_id: str, provider_id: str):
        self.agent_provider_overrides[agent_id] = provider_id

    def add_provider(self, provider_data: dict):
        provider_data["api_key"] = _resolve_env(provider_data.get("api_key", ""))
        provider = Provider.from_dict(provider_data)
        self.providers[provider.id] = provider
        self.save_providers()

    def remove_provider(self, provider_id: str):
        self.providers.pop(provider_id, None)
        self.save_providers()
