"""
Gradio UI — Multi-Agent Video Generation System.

Three tabs:
  1. Generate  — stepped pipeline: Planning → Generation → Composition
  2. Config    — per-agent Provider/Model selection
  3. Pipeline  — online YAML editor for providers.yaml + pipeline.yaml

Launch: python app.py
"""
import queue
import threading
import gradio as gr
from core.config_manager import ConfigManager
from core.pipeline import PipelineExecutor, PipelineSession

cfg = ConfigManager.get_instance()
executor = PipelineExecutor(cfg)


# ═══════════════════════════════════════════════════
#  Tab 1: Generate (stepped: Planning → Generation → Composition)
# ═══════════════════════════════════════════════════

def _normalize_images(style_refs, char_ref):
    """Normalize Gradio File/Image inputs to flat path lists."""
    if style_refs is None:
        style_images = []
    elif isinstance(style_refs, str):
        style_images = [style_refs]
    else:
        style_images = [f for f in style_refs if f is not None]
    char_images = [char_ref] if char_ref is not None else []
    return style_images, char_images


def _run_in_thread(fn, *args):
    """Run fn in a daemon thread; return (thread, queue) for polling."""
    q = queue.Queue()

    def wrapper():
        try:
            result = fn(*args, progress_callback=lambda m, d: q.put((m, d)))
            q.put(("done", result))
        except Exception as e:
            q.put(("error", str(e)))

    t = threading.Thread(target=wrapper, daemon=True)
    t.start()
    return t, q


def _drain_queue(q, display_logs):
    """Drain all available messages from queue. Non-blocking — returns immediately.

    Returns (result, error) where result is the 'done' payload or None.
    """
    result = None
    error = None
    while True:
        try:
            msg_type, data = q.get_nowait()
            if msg_type == "log":
                display_logs.append(data)
            elif msg_type == "agent_start":
                display_logs.append(f"  ⏳ {data} 执行中...")
            elif msg_type == "agent_done":
                for j in range(len(display_logs) - 1, -1, -1):
                    if f"⏳ {data}" in display_logs[j]:
                        display_logs[j] = f"  ✓ {data}"
                        break
            elif msg_type == "done":
                result = data
            elif msg_type == "error":
                error = data
                display_logs.append(f"\n[错误] {data}")
        except queue.Empty:
            break
    return result, error


# ── Step 1: Planning ──

def handle_planning(text_input, style_refs, char_ref, duration, session_state):
    """Run planning phase. Yields (storyboard, video, logs, session, gen_btn)."""
    if not text_input or not text_input.strip():
        yield {}, None, "[提示] 请输入视频描述。", session_state, gr.update(interactive=False)
        return

    style_images, char_images = _normalize_images(style_refs, char_ref)

    if duration and duration != 10:
        full_input = f"{text_input.strip()}，时长{duration}秒"
    else:
        full_input = text_input.strip()

    display_logs = ["⏳ 启动规划阶段..."]
    session = executor.start_session(full_input, style_images, char_images)

    thread, q = _run_in_thread(executor.run_planning, session)
    yield {}, None, "\n".join(display_logs), session, gr.update(interactive=False)

    import time
    result = None
    while thread.is_alive() or not q.empty():
        result, _err = _drain_queue(q, display_logs)
        if result:
            storyboard = result.context.get("scene_planner", {})
            display_logs.append(f"\n✓ 规划完成 — 输出目录: {result.run_dir}")
            yield (storyboard, None, "\n".join(display_logs),
                   result, gr.update(interactive=True))
            return
        yield ({}, None, "\n".join(display_logs), session,
               gr.update(interactive=False))
        time.sleep(0.3)

    # Thread ended without "done" — check for errors
    _drain_queue(q, display_logs)
    yield ({}, None, "\n".join(display_logs), session,
           gr.update(interactive=False))


# ── Step 2: Generation ──

def handle_generation(session_state):
    """Run generation phase. Yields (storyboard, video, logs, session, compose_btn)."""
    if session_state is None or session_state.phase != "planned":
        yield ({}, None, "[提示] 请先完成规划。",
               session_state, gr.update(interactive=False))
        return

    display_logs = list(session_state.logs)
    display_logs.append("⏳ 开始视频生成...")

    thread, q = _run_in_thread(executor.run_generation, session_state)
    yield (session_state.context.get("scene_planner", {}),
           None, "\n".join(display_logs),
           session_state, gr.update(interactive=False))

    import time
    result = None
    while thread.is_alive() or not q.empty():
        result, _err = _drain_queue(q, display_logs)
        if result:
            n_ok = sum(
                1 for s in result.final_shots if not s.get("error"))
            display_logs.append(
                f"\n✓ 生成完成 — {n_ok}/{len(result.final_shots)} 个镜头成功")
            has_ok = n_ok > 0
            yield (result.context.get("scene_planner", {}),
                   None, "\n".join(display_logs), result,
                   gr.update(interactive=has_ok))
            return
        yield (session_state.context.get("scene_planner", {}),
               None, "\n".join(display_logs), session_state,
               gr.update(interactive=False))
        time.sleep(0.3)

    _drain_queue(q, display_logs)
    yield (session_state.context.get("scene_planner", {}),
           None, "\n".join(display_logs), session_state,
           gr.update(interactive=False))


# ── Step 3: Composition ──

def handle_composition(session_state):
    """Run composition phase. Yields (storyboard, video, logs, session)."""
    if session_state is None or session_state.phase != "generated":
        yield ({}, None, "[提示] 请先完成视频生成。", session_state)
        return

    display_logs = list(session_state.logs)
    display_logs.append("⏳ 开始视频拼接...")

    thread, q = _run_in_thread(executor.run_composition, session_state)
    yield ({}, None, "\n".join(display_logs), session_state)

    import time
    result = None
    while thread.is_alive() or not q.empty():
        result, _err = _drain_queue(q, display_logs)
        if result:
            comp = result.composition
            video_path = (
                comp.get("output_path")
                if comp.get("status") != "error" else None)
            if video_path:
                display_logs.append(f"\n✓ 拼接完成 — {video_path}")
            yield ({}, video_path, "\n".join(display_logs), result)
            return
        yield ({}, None, "\n".join(display_logs), session_state)
        time.sleep(0.3)

    _drain_queue(q, display_logs)
    yield ({}, None, "\n".join(display_logs), session_state)


# ── Step 4: Reset ──

def handle_reset():
    """Reset all state for a new run."""
    return {}, None, "", None, gr.update(interactive=False), gr.update(interactive=False)


# ── Build UI ──

def build_generation_tab():
    gr.Markdown("## 视频生成（分步执行）")
    gr.Markdown(
        "**① 开始规划** → 预览分镜 → **② 生成视频** → 检查镜头 → **③ 拼接输出**")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="描述你想要的视频",
                placeholder="赛博朋克街景，主角在霓虹灯照耀的雨夜中行走，10秒",
                lines=4,
            )
            style_ref = gr.File(
                label="风格参考图（可多张拖入）",
                file_count="multiple", file_types=["image"],
                type="filepath")
            char_ref = gr.Image(label="角色参考图（可选）", type="filepath")
            duration = gr.Slider(5, 120, value=10, step=5,
                                 label="视频时长（秒）")

            with gr.Row():
                plan_btn = gr.Button("① 开始规划", variant="primary", size="lg")
                gen_btn = gr.Button("② 生成视频", variant="primary", size="lg",
                                    interactive=False)
                compose_btn = gr.Button("③ 拼接输出", variant="primary", size="lg",
                                        interactive=False)
            reset_btn = gr.Button("⟳ 重置", variant="secondary", size="sm")

        with gr.Column(scale=1):
            storyboard_out = gr.JSON(label="分镜表")
            video_out = gr.Video(label="生成的视频")
            log_out = gr.Textbox(label="执行日志", lines=12, max_lines=25,
                                 autoscroll=True)

    # ── Hidden session state ──
    session_state = gr.State(None)

    # ── Wire buttons ──
    plan_btn.click(
        fn=handle_planning,
        inputs=[text_input, style_ref, char_ref, duration, session_state],
        outputs=[storyboard_out, video_out, log_out, session_state, gen_btn],
    )

    gen_btn.click(
        fn=handle_generation,
        inputs=[session_state],
        outputs=[storyboard_out, video_out, log_out, session_state, compose_btn],
    )

    compose_btn.click(
        fn=handle_composition,
        inputs=[session_state],
        outputs=[storyboard_out, video_out, log_out, session_state],
    )

    reset_btn.click(
        fn=handle_reset,
        inputs=[],
        outputs=[storyboard_out, video_out, log_out, session_state,
                 gen_btn, compose_btn],
    )


# ═══════════════════════════════════════════════════
#  Tab 2: Agent Config
# ═══════════════════════════════════════════════════

def build_agent_config_tab():
    gr.Markdown("## Agent 配置")
    gr.Markdown("为每个 Agent 选择 Provider 和 Model。保存后立即生效。")

    agent_ids = cfg.get_agent_ids()
    provider_ids = list(cfg.providers.keys())
    rows = []

    for agent_id in agent_ids:
        agent_cfg = cfg.get_agent_config(agent_id)
        name = agent_cfg.get("name", agent_id)
        desc = agent_cfg.get("description", "")
        model_type = agent_cfg.get("model_type", "text")
        default_provider = agent_cfg.get("default_provider", "")
        current_provider = (
            cfg.agent_provider_overrides.get(agent_id) or default_provider)

        provider_choices = [
            f"{pid} — {cfg.providers[pid].name}" for pid in provider_ids]

        # Models for current provider, filtered by agent's model_type
        model_ids = []
        if current_provider in cfg.providers:
            p = cfg.providers[current_provider]
            model_ids = [m.id for m in p.models
                         if m.type == model_type]
        current_model = cfg.agent_model_overrides.get(
            agent_id) or cfg.get_model_for_agent(agent_id)

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(
                    f"**{name}**  \n*{desc}*  \n类型: `{model_type}`")
            with gr.Column(scale=1):
                pd = gr.Dropdown(
                    choices=provider_choices,
                    value=(
                        f"{current_provider} — {cfg.providers[current_provider].name}"
                        if current_provider in cfg.providers else None),
                    label="Provider",
                )
            with gr.Column(scale=2):
                md = gr.Dropdown(
                    choices=model_ids,
                    value=current_model if current_model in model_ids else None,
                    label="Model",
                )
        rows.append((agent_id, pd, md))

    save_btn = gr.Button("保存配置", variant="primary")
    save_msg = gr.Textbox(label="状态")

    def save_configs(*widget_values):
        updated = 0
        for i in range(len(agent_ids)):
            agent_id = agent_ids[i]
            prov_raw = widget_values[i * 3 + 1]
            model_val = widget_values[i * 3 + 2]

            # Parse provider id from "siliconflow — 硅基流动" format
            if prov_raw:
                prov_id = prov_raw.split(" — ")[0]
                if prov_id in cfg.providers:
                    cfg.set_agent_provider(agent_id, prov_id)
                    updated += 1
            if model_val:
                cfg.set_agent_model(agent_id, model_val)

        executor.invalidate()
        return f"已保存 {updated} 个 Agent 配置。立即生效。"

    widget_list = []
    for agent_id, pd, md in rows:
        widget_list.extend([gr.Textbox(value=agent_id, visible=False), pd, md])

    save_btn.click(fn=save_configs, inputs=widget_list, outputs=[save_msg])


# ═══════════════════════════════════════════════════
#  Tab 3: Pipeline Config
# ═══════════════════════════════════════════════════

def build_pipeline_tab():
    gr.Markdown("## 流水线配置")
    gr.Markdown("在线编辑 YAML 配置文件。保存后自动重载。")

    def _read_yaml(filename):
        path = cfg.config_dir / filename
        return path.read_text(encoding="utf-8") if path.exists() else f"# {filename} not found"

    def _save_yaml(filename, new_content):
        import yaml
        yaml.safe_load(new_content)  # Validate
        (cfg.config_dir / filename).write_text(new_content, encoding="utf-8")
        if filename == "providers.yaml":
            cfg._load_providers()
        elif filename == "pipeline.yaml":
            cfg._load_pipeline()
        executor.invalidate()

    # Pipeline editor
    gr.Markdown("### pipeline.yaml")
    pipeline_editor = gr.Code(
        value=_read_yaml("pipeline.yaml"), language="yaml",
        label="pipeline.yaml", lines=20)

    def save_pipeline(yaml_str):
        try:
            _save_yaml("pipeline.yaml", yaml_str)
            return "流水线已保存。"
        except Exception as e:
            return f"YAML 错误: {e}"

    gr.Button("保存流水线", variant="primary").click(
        fn=save_pipeline, inputs=[pipeline_editor],
        outputs=[gr.Textbox(label="状态")])

    # Provider editor
    gr.Markdown("### providers.yaml")
    provider_editor = gr.Code(
        value=_read_yaml("providers.yaml"), language="yaml",
        label="providers.yaml", lines=15)

    def save_providers(yaml_str):
        try:
            _save_yaml("providers.yaml", yaml_str)
            return "Provider 已保存。"
        except Exception as e:
            return f"YAML 错误: {e}"

    gr.Button("保存 Provider", variant="primary").click(
        fn=save_providers, inputs=[provider_editor],
        outputs=[gr.Textbox(label="状态")])


# ═══════════════════════════════════════════════════
#  Entry
# ═══════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="Multi-Agent 视频生成系统") as demo:
        gr.Markdown("# Multi-Agent 视频生成系统")
        gr.Markdown(
            f"流水线: **{cfg.get_pipeline_name()}** | "
            f"Agent: **{len(cfg.get_agent_ids())}** | "
            f"Provider: **{len(cfg.providers)}**"
        )

        with gr.Tabs():
            with gr.TabItem("生成视频"):
                build_generation_tab()
            with gr.TabItem("Agent 配置"):
                build_agent_config_tab()
            with gr.TabItem("流水线配置"):
                build_pipeline_tab()

    return demo


if __name__ == "__main__":
    import sys
    print(f"Python: {sys.version}")
    print(f"Pipeline: {cfg.get_pipeline_name()}")
    print(f"Agents: {cfg.get_agent_ids()}")
    print(f"Providers: {list(cfg.providers.keys())}")
    demo = build_app()
    demo.launch(share=False)
