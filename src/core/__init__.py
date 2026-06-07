"""
core/ — Decoupled architecture for the Multi-Agent video generation system.

Layers:
  provider.py      : Provider + ModelInfo — encapsulates API endpoints
  config_manager.py: ConfigManager — single source of truth for runtime config
  base_agent.py    : BaseAgent — receives model/client/tools via injection
  agent_factory.py : AgentFactory — creates agents from config
  pipeline.py      : PipelineExecutor — config-driven DAG execution

Note: PipelineExecutor is NOT re-exported here to avoid circular imports
(tools → config → core.config_manager → core.init → core.pipeline → tools).
Import it directly: from core.pipeline import PipelineExecutor
"""
from core.provider import Provider, ModelInfo
from core.config_manager import ConfigManager
from core.base_agent import BaseAgent
from core.agent_factory import register_agent

__all__ = [
    "Provider", "ModelInfo",
    "ConfigManager",
    "BaseAgent",
    "register_agent",
]
