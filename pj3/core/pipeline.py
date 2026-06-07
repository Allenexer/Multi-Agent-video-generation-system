"""
Configurable Pipeline Executor — YAML-driven DAG execution.

Three-phase pipeline:
  Planning:    Director → {StyleAnalyzer, ScriptWriter, CharacterDesigner}
               → ScenePlanner → PromptEngineer → ConsistencyGuard
  Generation:  CogVideoX t2v/i2v chain with anchor frames
  Composition: moviepy concat → final.mp4

Continuity: Shot 1 t2v → extract last frame → Shot 2..N i2v anchored.
Style/character prompt fragments + style_index routing preserved.
"""
import os
from core.config_manager import ConfigManager
from core.agent_factory import AgentFactory
from core.image_ref import (
    ImageRef, classify_images, describe_for_agent,
    get_by_type, get_by_label,
)
from tools import (
    CogVideoXTool, KeyframeExtractor, VideoConcatTool,
)
import agents  # noqa: F401 — triggers @register_agent decorators

VALID_AGENT_IDS = {
    "director", "style_analyzer", "script_writer", "character_designer",
    "scene_planner", "prompt_engineer", "consistency_guard", "critic",
}
IDEA2VIDEO_MIN_CHARS = 40
DEFAULT_MAX_RETRIES = 3
BASE_OUTPUT = "outputs"


class PipelineExecutor:

    def __init__(self, config: ConfigManager = None):
        self.config = config or ConfigManager.get_instance()
        self.factory = AgentFactory(self.config)
        self.video_gen = CogVideoXTool()
        self.extractor = KeyframeExtractor()
        self.concat = VideoConcatTool()
        self.run_dir = ""

    # ══════════════════════════════════════════════
    #  Validation
    # ══════════════════════════════════════════════

    @classmethod
    def is_valid_agent(cls, agent_id: str) -> bool:
        return agent_id in VALID_AGENT_IDS

    @classmethod
    def validate_stage(cls, stage: dict) -> list[str]:
        return [a for a in stage.get("agents", [])
                if not cls.is_valid_agent(a)]

    # ══════════════════════════════════════════════
    #  Main entry
    # ══════════════════════════════════════════════

    def run(self, user_input: str,
            style_images: list = None,
            character_images: list = None,
            progress_callback=None) -> dict:
        from datetime import datetime
        self.run_dir = os.path.join(
            BASE_OUTPUT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.run_dir, exist_ok=True)

        style_images = style_images or []
        character_images = character_images or []
        refs = classify_images(style_images, character_images)
        self._refs = refs
        img_desc = describe_for_agent(refs)
        all_paths = [r.path for r in refs]

        stages = self.config.get_pipeline_stages()
        context = {}
        logs = []
        errors = []

        def _log(msg):
            logs.append(msg)
            if progress_callback:
                progress_callback("log", msg)

        _log(f"[Run] 输出目录: {self.run_dir}")
        _log(f"[Run] 参考图片:\n{img_desc}")

        # ── Idea2Video ──
        expanded_input = user_input
        if len(user_input) < IDEA2VIDEO_MIN_CHARS:
            _log("[Idea2Video] 短输入，自动展开...")
            expanded_input = self._expand_idea(user_input, _log)

        # ── Planning phase ──
        for stage in stages:
            stage_name = stage.get("name", stage.get("id", ""))
            unknown = self.validate_stage(stage)
            if unknown:
                msg = f"[{stage_name}] 未知 Agent: {unknown}"
                _log(f"  ⚠ {msg}")
                errors.append(msg)

            _log(f"[{stage_name}] {len(stage.get('agents', []))} agents...")

            for agent_id in stage.get("agents", []):
                if not self.is_valid_agent(agent_id):
                    _log(f"  ⊘ {agent_id} — 跳过（未知）")
                    context[agent_id] = {"error": f"Unknown: {agent_id}",
                                          "skipped": True}
                    continue
                agent = self.factory.create(agent_id)
                if progress_callback:
                    progress_callback("agent_start", agent_id)
                try:
                    task = {
                        "user_input": expanded_input,
                        "stage": stage["id"],
                        "upstream": list(context.keys()),
                        "available_images": img_desc,
                    }
                    image_paths = (
                        all_paths
                        if agent.model_type == "vision" and all_paths
                        else None
                    )
                    result = agent.think(
                        task=task, context=context, image_paths=image_paths)
                    context[agent_id] = result
                    if progress_callback:
                        progress_callback("agent_done", agent_id)
                    else:
                        _log(f"  ✓ {agent_id}")
                except Exception as e:
                    err = f"{agent_id}: {type(e).__name__}: {e}"
                    _log(f"  ✗ {err}")
                    errors.append(err)
                    context[agent_id] = {"error": str(e)}

        # ── Multi-candidate selection ──
        self._select_best_candidates(context, logs)

        # ── Consistency anchors ──
        anchors = self._extract_anchors(context)

        # ── Generation phase ──
        storyboard = context.get("scene_planner", {})
        shots = storyboard.get("shots", [])
        final_shots = []
        composition = {}
        if shots:
            final_shots = self._generate_chain(
                shots, anchors, _log,
                max_retries=DEFAULT_MAX_RETRIES)
            comp_path = os.path.join(self.run_dir, "final.mp4")
            composition = self.compose(final_shots, output_path=comp_path)
            if composition.get("status") != "error":
                logs.append(
                    f"[Compose] 输出: {composition.get('output_path', '')}")

        # ── Save artifacts ──
        self._save_artifacts(storyboard, final_shots, logs, errors)

        return {
            "pipeline_name": self.config.get_pipeline_name(),
            "context": context,
            "shots": final_shots,
            "composition": composition,
            "run_dir": self.run_dir,
            "logs": logs,
            "errors": errors,
        }

    # ══════════════════════════════════════════════
    #  Consistency anchors
    # ══════════════════════════════════════════════

    def _extract_anchors(self, context: dict) -> dict:
        anchors = {
            "style_fragments": [],    # [fragment_0, fragment_1, ...] indexed
            "style_fragment": "",     # fallback single
            "character_fragment": "",
        }
        sa = context.get("style_analyzer", {})
        if isinstance(sa, dict) and not sa.get("error"):
            # Support multi-style: array of fragments keyed by index
            style_list = sa.get("style_fragments", [])
            if style_list:
                anchors["style_fragments"] = [
                    s.get("style_prompt_fragment", "") for s in style_list]
            anchors["style_fragment"] = sa.get("style_prompt_fragment", "")

        cd = context.get("character_designer", {})
        if isinstance(cd, dict) and not cd.get("error"):
            chars = cd.get("characters", [])
            if chars:
                anchors["character_fragment"] = chars[0].get(
                    "appearance_prompt_fragment", "")
            else:
                anchors["character_fragment"] = cd.get(
                    "appearance_prompt_fragment", "")
        return anchors

    # ══════════════════════════════════════════════
    #  Chain generation
    # ══════════════════════════════════════════════

    def _generate_chain(self, shots: list, anchors: dict,
                        _log, max_retries: int = 3) -> list:
        """t2v for Shot 1 and style-change shots; i2v anchored otherwise."""
        os.makedirs(self.run_dir, exist_ok=True)
        anchor_frame = None
        prev_style = None
        final_shots = []

        for i, shot in enumerate(shots):
            cur_style = shot.get("style_index") or 0

            # Break anchor chain on style change or gap
            if prev_style is not None and cur_style != prev_style:
                _log(f"  [Shot {shot.get('shot_id', i+1)}] 风格切换 {prev_style}→{cur_style}，重置锚定")
                anchor_frame = None
            sid = shot.get("shot_id", i + 1)
            base_prompt = shot.get("prompt", "")
            # Per-shot style routing
            style_idx = shot.get("style_index") or 0
            style_fragments = anchors.get("style_fragments", [])
            if style_fragments and 0 <= style_idx < len(style_fragments):
                style_part = style_fragments[style_idx]
            else:
                style_part = anchors.get("style_fragment", "")
            char_part = anchors.get("character_fragment", "")
            unified_prompt = (
                f"{style_part}, {char_part}, {base_prompt}".strip(", "))
            shot_dur = shot.get("duration_sec") or 5
            cog_dur = 5 if shot_dur <= 5 else 10

            _log(f"[Shot {sid}] prompt: {unified_prompt[:80]}...")

            had_reset = (anchor_frame is None)
            result = None
            for attempt in range(max_retries):
                try:
                    if i == 0 or anchor_frame is None:
                        _log(f"  -> t2v (CogVideoX) attempt {attempt + 1}")
                        _log("  ⏳ 提交 CogVideoX，等待生成...")
                        result = self.video_gen.generate_t2v(
                            prompt=unified_prompt)
                    else:
                        _log(f"  -> i2v (CogVideoX, anchored) attempt {attempt + 1}")
                        _log("  ⏳ 提交 CogVideoX，等待生成...")
                        result = self.video_gen.generate_i2v(
                            image_path=anchor_frame,
                            prompt=unified_prompt)

                    if result.get("video_url"):
                        _log(f"  ✓ 生成成功: {result['video_url']}")
                        break
                except Exception as e:
                    _log(f"  ✗ 尝试 {attempt + 1} 失败: {e}")
                    if attempt < max_retries - 1:
                        unified_prompt += (
                            ", high quality, detailed, consistent style")
                    result = None

            if result is None:
                _log(f"  ✗ Shot {sid} 重试耗尽，跳过")
                final_shots.append({"shot_id": sid, "error": "All retries exhausted"})
                anchor_frame = None
                continue

            video_url = result.get("video_url", "")

            # Extract anchor for next shot
            if video_url and i < len(shots) - 1:
                try:
                    local = self.extractor.download_video(video_url)
                    anchor_frame = self.extractor.extract_last_frame(local)
                    _log("  ✓ 锚定帧已提取")
                except Exception as e:
                    _log(f"  ⚠ 锚定提取失败: {e}")
                    anchor_frame = None

            gen_method = "t2v" if (i == 0 or had_reset) else "i2v"
            final_shots.append({
                "shot_id": sid,
                "video_url": video_url,
                "generation_method": gen_method,
            })
            prev_style = cur_style

        return final_shots

    # ══════════════════════════════════════════════
    #  Composition
    # ══════════════════════════════════════════════

    def compose(self, shots: list, output_path: str = None) -> dict:
        clip_paths = []
        for shot in shots:
            url = shot.get("video_url", "")
            if url and not shot.get("error"):
                try:
                    local = KeyframeExtractor.download_video(url)
                    clip_paths.append(local)
                except Exception:
                    continue

        if not clip_paths:
            return {"status": "error", "message": "没有可拼接的视频片段"}

        if output_path is None:
            output_path = os.path.join(
                self.run_dir or BASE_OUTPUT,
                f"final_{len(clip_paths)}shots.mp4")
        return self.concat.execute(
            clip_paths=clip_paths, output_path=output_path)

    # ══════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════

    def _expand_idea(self, short_input: str, log_fn) -> str:
        try:
            director = self.factory.create("director")
            saved = director.role_prompt
            director.role_prompt = (
                "用户给了一个很短的视频创意。请扩展为一段详细的视频脚本描述"
                "（100-200字），包含场景、氛围、视觉风格、角色动作、叙事节奏。"
                "只输出扩展后的中文描述，不要 JSON。"
            )
            result = director.think(
                task={"user_input": short_input}, use_cache=True)
            director.role_prompt = saved
            expanded = result.get("summary", "") or str(result)
            if len(expanded) > len(short_input) * 2:
                log_fn(
                    f"  ✓ 展开: {short_input[:30]}... -> {expanded[:60]}...")
                return expanded
        except Exception as e:
            log_fn(f"  ⚠ Idea2Video 展开失败: {e}")
        return short_input

    def _select_best_candidates(self, context: dict, logs: list):
        pe = context.get("prompt_engineer", {})
        candidates = pe.get("candidates", [])
        if len(candidates) <= 1:
            return
        best_idx = pe.get("recommended_index", 0)
        if 0 <= best_idx < len(candidates):
            best = candidates[best_idx]
            logs.append(
                f"[Multi-Candidate] 选定 #{best_idx}: "
                f"'{best.get('prompt', '')[:60]}...'")
        context["prompt_engineer"]["selected_candidate_index"] = best_idx
        context["prompt_engineer"]["candidates"] = [candidates[best_idx]]

    def _save_artifacts(self, storyboard: dict, shots: list,
                        logs: list, errors: list):
        import json

        path = os.path.join(self.run_dir, "storyboard.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(storyboard, f, ensure_ascii=False, indent=2)

        path = os.path.join(self.run_dir, "logs.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(logs))
            if errors:
                f.write(f"\n\n=== {len(errors)} ERRORS ===\n")
                for e in errors:
                    f.write(f"  {e}\n")

        summary = [{"shot_id": s.get("shot_id", "?"),
                     "method": s.get("generation_method", "?"),
                     "video_url": s.get("video_url", ""),
                     "error": s.get("error", "")}
                   for s in shots]
        path = os.path.join(self.run_dir, "shots.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def invalidate(self):
        self.factory.invalidate()
