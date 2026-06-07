# Multi-Agent Video Generation System

A configurable multi-agent system that automates video production — from scriptwriting to final composition — using 7 specialized AI agents.

## Quick Start

```bash
cd pj3
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
python test_agent.py    # verify
python app.py           # launch UI
```

Open `http://127.0.0.1:7860`.

## Architecture

```
config/     YAML-driven configuration (providers, agents, pipeline)
core/       Engine: Agent base class, factory, pipeline executor, image ref system
agents/     Specialized agent implementations
memory/     Response cache + character/style persistence
tools/      Video generation, image analysis, frame extraction, composition
app.py      Gradio UI (3 tabs: Generate | Config | Pipeline)
```

## API Keys Required

| Platform | Key |
|----------|-----|
| [SiliconFlow](https://siliconflow.cn) | `SILICONFLOW_API_KEY` |
| [Zhipu AI](https://open.bigmodel.cn) | `ZHIPU_API_KEY` |
| [Volcengine](https://console.volcengine.com) (optional) | `ARK_API_KEY` |

## Pipeline

```
Director → {StyleAnalyzer, ScriptWriter, CharacterDesigner}
         → ScenePlanner → PromptEngineer → ConsistencyGuard
         → CogVideoX-3 (t2v → i2v anchor chain)
         → moviepy concat → final.mp4
```
