# Multi-Agent Video Generation System

A configurable multi-agent system that automates video production — from scriptwriting to final composition — using 7 specialized AI agents with an agentic feedback loop.

## Quick Start

```bash
cd pj3
pip install -r requirements.txt
pip install 'volcengine-python-sdk[ark]'  # Seedance 2.0 SDK
cp .env.example .env   # fill in your API keys
python app.py           # launch UI
```

Open `http://127.0.0.1:7860`.

## Architecture

```
config/     YAML-driven configuration (providers, agents, pipeline)
core/       Engine: Agent base class, factory, pipeline executor with feedback loop
agents/     Specialized agent implementations (StyleAnalyzer + KMeans)
memory/     Response cache (5-min TTL)
tools/      Seedance 2.0 video gen, Seedream 5.0 keyframe gen, image analysis, composition
app.py      Gradio UI (3 tabs: Generate | Config | Pipeline)
```

## API Keys Required

| Platform | Key |
|----------|-----|
| [SiliconFlow](https://siliconflow.cn) | `SILICONFLOW_API_KEY` |
| [Zhipu AI](https://open.bigmodel.cn) | `ZHIPU_API_KEY` |
| [Volcengine Ark](https://console.volcengine.com) | `ARK_API_KEY` |

## Pipeline

```
Planning Phase:
  Director → {StyleAnalyzer, ScriptWriter, CharacterDesigner}
           → ScenePlanner → PromptEngineer → ConsistencyGuard
           ⇄ ConsistencyGuard feedback loop (auto-correction, up to 3 rounds)

Generation Phase (per segment):
  shot_change=true  → Seedream 5.0 start keyframe → user selects
                    → Seedance 2.0 i2v (start frame + char portraits + style refs)
  shot_change=false → Seedance 2.0 i2v (prev video last frame + char portraits)

Composition:
  moviepy concat → final.mp4
```

## Key Features

- **Agentic Feedback Loop**: ConsistencyGuard evaluates → feeds issues back to ScenePlanner for auto-correction
- **Shot-Change Continuity**: Segments default to continuous (prev video last frame inherited); only explicit shot changes trigger fresh keyframe generation
- **Character Portraits**: Pre-generated for each character × style combination, ensuring identity consistency across styles
- **Typed Reference Images**: Seedance 2.0 receives semantic roles (start frame, character, style) rather than monolithic image arrays
- **Human-in-the-Loop**: Per-segment approval with prompt editing, keyframe selection, and regeneration

## Models

| Role | Provider | Model |
|------|----------|-------|
| Text Agents (×6) | SiliconFlow | DeepSeek-V3.2 |
| Visual Agents (×2) | Zhipu AI | GLM-5V-Turbo |
| Keyframe Generation | Volcengine Ark | Seedream 5.0 |
| Video Generation | Volcengine Ark | Seedance 2.0 |
