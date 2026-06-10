"""
Gradio UI — Multi-Agent Video Generation System.

Three tabs:
  1. Generate  — stepped pipeline: Planning → Generation → Composition
  2. Config    — per-agent Provider/Model selection
  3. Pipeline  — online YAML editor for providers.yaml + pipeline.yaml

Launch: python app.py
"""
import os
import queue
import threading
import gradio as gr
from core.config_manager import ConfigManager
from core.pipeline import PipelineExecutor, PipelineSession

cfg = ConfigManager.get_instance()
executor = PipelineExecutor(cfg)


# ═══════════════════════════════════════════════════
#  Tab 1: Generate (Planning → Keyframes → Video → Compose)
# ═══════════════════════════════════════════════════

def _normalize_images(style_refs, char_ref):
    if style_refs is None:
        return [], []
    if isinstance(style_refs, str):
        style_images = [style_refs]
    else:
        style_images = [f for f in style_refs if f is not None]
    if char_ref is None:
        char_images = []
    elif isinstance(char_ref, str):
        char_images = [char_ref]
    else:
        char_images = [f for f in char_ref if f is not None]
    return style_images, char_images


def _run_in_thread(fn, *args):
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


def _current_seg(session_state):
    """Get current segment being worked on."""
    if session_state is None:
        return None
    si = getattr(session_state, 'gen_index', 0)
    segs = session_state.shots
    return segs[si] if segs and si < len(segs) else None


# ── Step 1: Planning ──

def handle_planning(text_input, style_refs, char_ref, duration, session_state):
    """Yields (storyboard, video, logs, session, prompt_box, kf_btn)."""
    if not text_input or not text_input.strip():
        yield ({}, None, "[提示] 请输入视频描述。", session_state,
               gr.update(value=""), gr.update(interactive=False))
        return

    style_images, char_images = _normalize_images(style_refs, char_ref)
    full_input = text_input.strip()
    if duration and duration != 10:
        full_input = f"{full_input}，时长{duration}秒"

    display_logs = ["⏳ 启动规划阶段..."]
    session = executor.start_session(full_input, style_images, char_images)

    thread, q = _run_in_thread(executor.run_planning, session)
    yield ({}, None, "\n".join(display_logs), session,
           gr.update(value=""), gr.update(interactive=False))

    import time
    result = None
    while thread.is_alive() or not q.empty():
        result, _err = _drain_queue(q, display_logs)
        if result:
            storyboard = result.context.get("scene_planner", {})
            seg = _current_seg(result)
            prompt_val = seg.get("prompt", "") if seg else ""
            display_logs.append(f"\n✓ 规划完成 — 输出目录: {result.run_dir}")
            yield (storyboard, None, "\n".join(display_logs),
                   result, gr.update(value=prompt_val),
                   gr.update(interactive=True))
            return
        yield ({}, None, "\n".join(display_logs), session,
               gr.update(value=""), gr.update(interactive=False))
        time.sleep(0.3)

    _drain_queue(q, display_logs)
    yield ({}, None, "\n".join(display_logs), session,
           gr.update(value=""), gr.update(interactive=False))


# ── Step 2: Start keyframe generation ──

def handle_start_frames(session_state, prompt_text):
    """Generate START keyframe candidates. Yields (logs, gallery, paths, video_btn, start_select)."""
    if session_state is None:
        yield ("[提示] 请先完成规划。", None, [], gr.update(interactive=False), None)
        return

    si = getattr(session_state, 'gen_index', 0)
    seg = _current_seg(session_state)
    if seg is None:
        yield ("[提示] 当前段不存在。", None, [], gr.update(interactive=False), None)
        return

    if prompt_text and prompt_text.strip():
        seg["prompt"] = prompt_text.strip()

    display_logs = list(session_state.logs)
    display_logs.append(f"⏳ Seedream 生成段{si+1}起始帧 (切镜)...")

    thread, q = _run_in_thread(
        executor.generate_keyframes, session_state, si, 4)
    yield ("\n".join(display_logs), None, [], gr.update(interactive=False), None)

    import time
    result = None
    while thread.is_alive() or not q.empty():
        result, _err = _drain_queue(q, display_logs)
        if result is not None and isinstance(result, list) and result:
            display_logs.append(f"✓ 已生成 {len(result)} 张候选起始帧")
            yield ("\n".join(display_logs),
                   gr.update(value=result, interactive=True), result,
                   gr.update(interactive=True),
                   result[0])
            return
        yield ("\n".join(display_logs), gr.update(value=None, interactive=False),
               [], gr.update(interactive=False), None)
        time.sleep(0.3)

    _drain_queue(q, display_logs)
    yield ("\n".join(display_logs), None, [],
           gr.update(interactive=False), None)


# ── Step 3: Video via i2v ──

def handle_video_keyframe(session_state, start_frame):
    """Generate video via i2v from start frame. Yields (logs, session, video, next_btn, compose_btn)."""
    if session_state is None or not start_frame:
        yield ("[提示] 请先生成起始帧。", session_state,
               None, gr.update(), gr.update(interactive=False))
        return

    si = getattr(session_state, 'gen_index', 0)
    segs = session_state.shots
    display_logs = list(session_state.logs)
    display_logs.append(f"⏳ i2v 锚定链生成段 {si+1} 视频...")

    sf = start_frame
    if isinstance(start_frame, str) and "," in start_frame:
        sf = start_frame.split(",")[0].strip()

    display_logs.append(f"[Debug] start_frame={sf}")
    display_logs.append(f"[Debug] exists={os.path.exists(sf) if sf else 'None'}")

    thread, q = _run_in_thread(
        executor.generate_video_with_keyframe, session_state, si, sf)
    yield ("\n".join(display_logs), session_state,
           None, gr.update(), gr.update(interactive=False))

    import time
    result = None
    while thread.is_alive() or not q.empty():
        result, _err = _drain_queue(q, display_logs)
        if result and isinstance(result, dict):
            video_path = result.get("local_video")
            err = result.get("error")
            if video_path:
                display_logs.append(f"✓ 段 {si+1} 视频生成完成")
                # Replace existing entry for this segment (supports regeneration)
                seg_entry = {
                    "segment_id": si + 1,
                    "video_url": result.get("video_url", ""),
                    "local_video": video_path,
                    "generation_method": "i2v_anchor",
                }
                # Remove old entry if present, then append
                session_state.final_shots = [
                    s for s in session_state.final_shots
                    if s.get("segment_id") != si + 1]
                session_state.final_shots.append(seg_entry)
                session_state.logs = display_logs
                # Don't advance gen_index here — user may want to regenerate
                is_last = (si + 1 >= len(segs))
                session_state.gen_index = si  # stay on current segment
                session_state.phase = "generated" if is_last else "generating"

                if is_last:
                    yield ("\n".join(display_logs), session_state,
                           video_path,
                           gr.update(interactive=False, value="✓ 全部完成"),
                           gr.update(interactive=True))
                else:
                    yield ("\n".join(display_logs), session_state,
                           video_path,
                           gr.update(interactive=True,
                                     value=f"✓ 完成，点▶进入下一段 ({si+2}/{len(segs)})"),
                           gr.update(interactive=False))
                return
            elif err:
                display_logs.append(f"✗ 视频生成失败: {err}")
        yield ("\n".join(display_logs), session_state,
               None, gr.update(), gr.update(interactive=False))
        time.sleep(0.3)

    _drain_queue(q, display_logs)
    yield ("\n".join(display_logs), session_state,
           None, gr.update(), gr.update(interactive=False))


# ── Step 4: Next segment trigger ──

def handle_next_segment(session_state):
    """Advance to next segment.
    - shot_change=false + prev_end exists → auto-inherit as start, enable video
    - shot_change=true → enable start_btn for Seedream generation
    Returns (prompt, gallery, paths, select, start_btn, video_btn, compose_btn, next_btn).
    """
    if session_state is None:
        return (gr.update(), gr.update(value=None, interactive=False), [], None,
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update())

    # Advance to next segment (user confirmed current one)
    session_state.gen_index = getattr(session_state, 'gen_index', 0) + 1
    seg = _current_seg(session_state)
    if seg is None:
        # Already at last segment — nothing to advance to
        return (gr.update(), gr.update(value=None, interactive=False), [], None,
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=True),  # compose ready
                gr.update(interactive=False))
    prompt_val = seg.get("prompt", "")
    shot_change = seg.get("shot_change", False) if seg else False
    prev_end = getattr(session_state, 'last_end_frame', '')

    if not shot_change and prev_end:
        # Continuity: auto-inherit prev video last frame, skip to video
        return (gr.update(value=prompt_val),
                gr.update(value=[prev_end], interactive=True), [prev_end], prev_end,
                gr.update(interactive=False),   # start_btn: not needed
                gr.update(interactive=True),    # video_btn: ready
                gr.update(interactive=False),
                gr.update())
    else:
        # Shot change: need fresh Seedream start frame
        return (gr.update(value=prompt_val),
                gr.update(value=None, interactive=False), [], None,
                gr.update(interactive=True),    # start_btn: generate
                gr.update(interactive=False),   # video_btn: wait
                gr.update(interactive=False),
                gr.update())


# ── Step 5: Composition ──

def handle_composition(session_state):
    if session_state is None or session_state.phase != "generated":
        yield ({}, None, "[提示] 请先完成全部视频生成。", session_state)
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
            video_path = (comp.get("output_path")
                          if comp.get("status") != "error" else None)
            if video_path:
                display_logs.append(f"\n✓ 拼接完成 — {video_path}")
            yield ({}, video_path, "\n".join(display_logs), result)
            return
        yield ({}, None, "\n".join(display_logs), session_state)
        time.sleep(0.3)

    _drain_queue(q, display_logs)
    yield ({}, None, "\n".join(display_logs), session_state)


def handle_reset():
    return ({}, None, "", None, gr.update(value=""),
            gr.update(value=None, interactive=False), [],
            None,
            gr.update(interactive=False), gr.update(interactive=False),
            gr.update(interactive=False), gr.update(interactive=False))


# ── Build UI ──

def build_generation_tab():
    gr.Markdown("## 视频生成")
    gr.Markdown("**①规划** → 起始帧 → **②结束帧** → **③首尾帧视频** → 下一段")

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="描述你想要的视频",
                placeholder="赛博朋克街景，主角在霓虹灯照耀的雨夜中行走，10秒",
                lines=4)
            style_ref = gr.File(
                label="风格参考图（可多张拖入）",
                file_count="multiple", file_types=["image"],
                type="filepath")
            char_ref = gr.File(
                label="角色参考图（可多张拖入，顺序=角色0/1/2）",
                file_count="multiple", file_types=["image"],
                type="filepath")
            duration = gr.Slider(5, 120, value=10, step=5,
                                 label="视频时长（秒）")

            with gr.Row():
                plan_btn = gr.Button("① 开始规划", variant="primary")
                start_btn = gr.Button("② 生成起始帧（切镜时）", variant="primary",
                                      interactive=False)
                video_btn = gr.Button("③ 生成视频", variant="primary",
                                      interactive=False)
            with gr.Row():
                next_btn = gr.Button("▶ 下一段", variant="secondary",
                                     interactive=False)
                compose_btn = gr.Button("④ 拼接输出", variant="primary",
                                        interactive=False)
            reset_btn = gr.Button("⟳ 重置", size="sm")

        with gr.Column(scale=2):
            storyboard_out = gr.JSON(label="分镜表")
            prompt_editor = gr.Textbox(
                label="当前段 Prompt（可编辑）", lines=2,
                placeholder="规划完成后显示...")
            start_gallery = gr.Gallery(
                label="起始帧候选（切镜时点击选择，连续段自动加载末帧）", columns=4, height=180,
                interactive=False, allow_preview=True,
                object_fit="contain")
            video_out = gr.Video(label="生成的视频")
            log_out = gr.Textbox(label="执行日志", lines=5, max_lines=15,
                                 autoscroll=True)

    # ── Hidden state ──
    session_state = gr.State(None)
    start_select = gr.State(None)
    start_paths = gr.State([])

    def _on_start_select(evt: gr.SelectData, paths):
        return paths[evt.index] if paths and evt.index < len(paths) else None

    start_gallery.select(
        fn=_on_start_select, inputs=[start_paths], outputs=[start_select])

    # ── Wire buttons ──
    plan_btn.click(
        fn=handle_planning,
        inputs=[text_input, style_ref, char_ref, duration, session_state],
        outputs=[storyboard_out, video_out, log_out, session_state,
                 prompt_editor, start_btn],
    )

    start_btn.click(
        fn=handle_start_frames,
        inputs=[session_state, prompt_editor],
        outputs=[log_out, start_gallery, start_paths, video_btn, start_select],
    )

    video_btn.click(
        fn=handle_video_keyframe,
        inputs=[session_state, start_select],
        outputs=[log_out, session_state, video_out, next_btn, compose_btn],
    )

    next_btn.click(
        fn=handle_next_segment,
        inputs=[session_state],
        outputs=[prompt_editor, start_gallery, start_paths, start_select,
                 start_btn, video_btn, compose_btn, next_btn],
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
                 prompt_editor, start_gallery, start_paths,
                 start_select, start_btn, video_btn,
                 compose_btn, next_btn],
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
