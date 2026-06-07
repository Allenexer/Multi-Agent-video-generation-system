"""
agents/ — Agent registration and convenience exports.

Importing this package triggers @register_agent decorators,
which register specialized agent classes with AgentFactory.

To add a new specialized agent:
  1. Create agents/my_agent.py with @register_agent("agent_id") class
  2. Import it below
"""
from core.base_agent import BaseAgent
from core.agent_factory import register_agent

# Import specialized agents — the @register_agent decorator
# on each class auto-registers it with AgentFactory.
import agents.style_analyzer  # noqa: F401 — triggers @register_agent

__all__ = ["BaseAgent", "register_agent"]
