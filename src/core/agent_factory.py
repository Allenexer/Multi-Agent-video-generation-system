"""
Agent factory — creates agents from config, with support for
specialized agent classes that have custom tools or behavior.
"""
from core.base_agent import BaseAgent
from core.config_manager import ConfigManager
from memory.cache import ResponseCache

# Registry of specialized agent classes (agent_id → class)
_SPECIALIZED: dict[str, type] = {}

# Shared cache across all agents (5-min TTL by default)
_SHARED_CACHE = ResponseCache(ttl_seconds=300)


def register_agent(agent_id: str):
    """Decorator to register a specialized agent class."""
    def decorator(cls):
        _SPECIALIZED[agent_id] = cls
        return cls
    return decorator


def set_cache_ttl(seconds: int):
    """Adjust cache TTL at runtime."""
    global _SHARED_CACHE
    _SHARED_CACHE = ResponseCache(ttl_seconds=seconds)


class AgentFactory:
    """
    Creates agent instances from configuration.

    All agents share a ResponseCache to reduce duplicate API calls.
    Specialized classes registered via @register_agent take precedence.
    """

    def __init__(self, config: ConfigManager = None):
        self.config = config or ConfigManager.get_instance()
        self._cache: dict[str, BaseAgent] = {}

    def create(self, agent_id: str) -> BaseAgent:
        if agent_id in self._cache:
            return self._cache[agent_id]

        agent_cfg = self.config.get_agent_config(agent_id)
        if not agent_cfg:
            raise KeyError(f"Unknown agent: {agent_id}")

        client = self.config.get_client_for_agent(agent_id)
        model = self.config.get_model_for_agent(agent_id)
        prompt = agent_cfg.get("prompt", "")
        model_type = agent_cfg.get("model_type", "text")

        # ── Tool registration for specialized agents ──
        tools = []
        if agent_id == "style_analyzer":
            from tools.image_analysis import ImageAnalysisTool
            tools.append(ImageAnalysisTool())

        cls = _SPECIALIZED.get(agent_id, BaseAgent)
        agent = cls(
            name=agent_id,
            role_prompt=prompt,
            client=client,
            model=model,
            model_type=model_type,
            cache=_SHARED_CACHE,
            tools=tools,
        )
        self._cache[agent_id] = agent
        return agent

    def invalidate(self):
        self._cache.clear()
        _SHARED_CACHE.clear()
