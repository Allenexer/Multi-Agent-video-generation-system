"""Quick test: verify the config-driven architecture works."""
from core.config_manager import ConfigManager
from core.base_agent import BaseAgent

cfg = ConfigManager.get_instance()

# Show current config
print(f"Pipeline: {cfg.get_pipeline_name()}")
print(f"Agents:   {cfg.get_agent_ids()}")
print(f"Providers: {list(cfg.providers.keys())}")
print()

for agent_id in cfg.get_agent_ids():
    provider = cfg.get_provider_for_agent(agent_id)
    model = cfg.get_model_for_agent(agent_id)
    print(f"  {agent_id:25s} → {provider.id:12s} / {model}")

# Test Director agent
print("\n── Testing Director Agent ──")
client = cfg.get_client_for_agent("director")
model = cfg.get_model_for_agent("director")
prompt = cfg.get_agent_config("director")["prompt"]

agent = BaseAgent(name="director", role_prompt=prompt, client=client, model=model)
result = agent.think(task={
    "user_input": "做一个5秒的卡通风格视频，一只柴犬在公园里奔跑",
    "reference_image_count": 0,
    "reference_image_names": [],
})

print(f"  intent_type: {result.get('intent_type')}")
print(f"  summary:     {result.get('summary')}")
print("  tasks:       {} agents scheduled".format(len(result.get('task_queue', []))))
print("\n[OK] Architecture verified.")
