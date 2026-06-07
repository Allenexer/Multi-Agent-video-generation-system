"""
Configurable Pipeline Executor — YAML-driven DAG execution.

Three-phase pipeline (supports both one-shot and stepped execution):
  Planning:    Director → {StyleAnalyzer, ScriptWriter, CharacterDesigner}
               → ScenePlanner → PromptEngineer → ConsistencyGuard
  Generation:  CogVideoX t2v / i2v anchor chain (text-level style control)
  Composition: moviepy concat → final.mp4

Stepped mode:  start_session() → run_planning() → run_generation()
               → run_composition(). Each step saves its artifacts and
               waits for user approval before proceeding.

Style control:   Text-only — StyleAnalyzer output fused into prompt.
                 CogVideoX image_url is NOT used for style reference
                 (i2v = "animate this image", not "generate in this style").
Continuity:      Shot 1 t2v → extract last frame → Shot 2..N i2v anchored.
Style change:    Reset anchor → t2v fresh start.
Keyframe mode:   Reserved for Seedance-generated start/end frames (future).
"""
import os
import json
import shutil
from dataclasses import dataclass, field

from core.config_manager import ConfigManager
from core.agent_factory import AgentFactory
from core.image_ref import (
    ImageRef, classify_images, describe_for_agent,
    get_by_type, get_by_label, get_style_by_index,
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


def _prompt_contains(longer: str, fragment: str, threshold: float = 0.4) -> bool:
    """Check if `longer` already covers a significant portion of `fragment`.

    Used to avoid prepending style/character fragments that the shot prompt
    already describes. Returns True if enough words from fragment appear in longer.
    """
    if not fragment or not longer:
        return False
    frag_words = set(w.lower().strip(",.;:!?") for w in fragment.split()
                     if len(w) > 3)
    if not frag_words:
        return False
    long_words = longer.lower().split()
    overlap = sum(1 for w in frag_words if w in long_words)
    return (overlap / len(frag_words)) >= threshold


@dataclass
class PipelineSession:
    """State object passed between stepped pipeline phases.

    Holds all intermediate results so the UI can pause between phases.
    Phase progression: init → planned → generated → composed.
    """
    run_dir: str = ""
    user_input: str = ""
    expanded_input: str = ""
    style_images: list = field(default_factory=list)
    character_images: list = field(default_factory=list)
    refs: list = field(default_factory=list)        # list[ImageRef]
    context: dict = field(default_factory=dict)     # agent outputs
    anchors: dict = field(default_factory=dict)     # style/char anchors
    shots: list = field(default_factory=list)       # planned shots (storyboard)
    final_shots: list = field(default_factory=list) # generated shot results
    composition: dict = field(default_factory=dict) # concat result
    logs: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    phase: str = "init"  # "init" | "planned" | "generated" | "composed"


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
        """One-shot full pipeline (backward-compatible).

        For stepped execution with approval gates, use:
          session = executor.start_session(...)
          session = executor.run_planning(session, progress_cb)
          session = executor.run_generation(session, progress_cb)
          session = executor.run_composition(session, progress_cb)
        """
        session = self.start_session(
            user_input, style_images, character_images)
        session = self.run_planning(session, progress_callback)
        session = self.run_generation(session, progress_callback)
        session = self.run_composition(session, progress_callback)
        return {
            "pipeline_name": self.config.get_pipeline_name(),
            "context": session.context,
            "shots": session.final_shots,
            "composition": session.composition,
            "run_dir": session.run_dir,
            "logs": session.logs,
            "errors": session.errors,
        }

    # ══════════════════════════════════════════════
    #  Consistency anchors
    # ══════════════════════════════════════════════

    def _extract_anchors(self, context: dict) -> dict:
        anchors = {
            "style_fragments": [],      # [fragment_0, fragment_1, ...] indexed
            "style_fragment": "",       # fallback single
            "character_fragment": "",
            "style_ref_images": [],     # per-style-index ref image paths
            "style_ref_image": None,    # fallback: first style ref image path
        }

        # ── Style reference image for pixel-level injection ──
        style_refs = get_by_type(getattr(self, '_refs', []), "style")
        if style_refs:
            anchors["style_ref_images"] = [r.path for r in style_refs]
            anchors["style_ref_image"] = style_refs[0].path

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
        """t2v / i2v anchor chain (no pixel-level style injection).

        Style is controlled via TEXT prompt injection only — the
        StyleAnalyzer's output is already fused into unified_prompt.
        CogVideoX i2v is reserved for TEMPORAL continuity (previous
        shot's last frame), NOT for style reference images.

        Generation strategy (per shot):
          - Shot 1 / style-reset:  t2v (pure text-to-video)
          - Shot 2..N with anchor: i2v (anchored to previous shot's last frame)
          - Future: first-last-frame when Seedance provides real keyframes
        """
        os.makedirs(self.run_dir, exist_ok=True)
        videos_dir = os.path.join(self.run_dir, "videos")
        frames_dir = os.path.join(self.run_dir, "frames")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)

        anchor_frame = None
        prev_style = None
        final_shots = []

        for i, shot in enumerate(shots):
            sid = shot.get("shot_id", i + 1)
            cur_style = shot.get("style_index") or 0

            # Break anchor chain on style change
            if prev_style is not None and cur_style != prev_style:
                _log(f"  [Shot {sid}] 风格切换 {prev_style}→{cur_style}，重置锚定")
                anchor_frame = None

            # ── Build unified prompt (text-level style + character + shot) ──
            base_prompt = shot.get("prompt", "")
            chars_in_shot = shot.get("characters_in_shot") or []
            style_idx = shot.get("style_index")

            # Resolve per-shot style fragment
            style_fragments = anchors.get("style_fragments", [])
            if style_fragments and style_idx is not None and 0 <= style_idx < len(style_fragments):
                style_part = style_fragments[style_idx]
            else:
                style_part = anchors.get("style_fragment", "")
            char_part = anchors.get("character_fragment", "")

            # ── Conditional concatenation ──
            # - Transition shots (no characters, no style_index) → base only
            # - Shots with characters but base already describes them → skip char
            # - Shots with style but base already describes it → skip style
            parts = []

            # Style: skip if style_index is null (transition) or base already contains it
            if style_idx is not None and style_part:
                if not _prompt_contains(base_prompt, style_part):
                    parts.append(style_part)

            # Character: skip if no characters in shot or base already contains it
            if chars_in_shot and char_part:
                if not _prompt_contains(base_prompt, char_part):
                    parts.append(char_part)

            parts.append(base_prompt)
            unified_prompt = ", ".join(parts)

            _log(f"[Shot {sid}] style_idx={style_idx}"
                 f" chars={chars_in_shot}"
                 f" prompt: {unified_prompt[:120]}...")

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
                        _log(f"  -> i2v (CogVideoX, anchored)"
                             f" attempt {attempt + 1}")
                        _log("  ⏳ 提交 CogVideoX，等待生成...")
                        result = self.video_gen.generate_i2v(
                            image_path=anchor_frame,
                            prompt=unified_prompt)

                    if result and result.get("video_url"):
                        _log(f"  ✓ 生成成功: {result['video_url']}")
                        break
                    if result and result.get("error"):
                        _log(f"  ✗ 尝试 {attempt + 1} 失败: {result['error']}")
                    elif not result:
                        _log(f"  ✗ 尝试 {attempt + 1} 返回空结果")
                except Exception as e:
                    _log(f"  ✗ 尝试 {attempt + 1} 异常: {e}")
                    if attempt < max_retries - 1:
                        unified_prompt += (
                            ", high quality, detailed, consistent style")
                    result = None

            if result is None:
                _log(f"  ✗ Shot {sid} 重试耗尽，跳过")
                final_shots.append(
                    {"shot_id": sid, "error": "All retries exhausted"})
                anchor_frame = None
                continue

            video_url = result.get("video_url", "")

            # ── Persist video + extract anchor frame for next shot ──
            local_video = None
            if video_url:
                try:
                    local_video = self.extractor.download_video(video_url)
                    # Save video to outputs
                    saved_video = os.path.join(
                        videos_dir, f"shot_{sid:02d}.mp4")
                    import shutil
                    shutil.copy(local_video, saved_video)
                    _log(f"  ✓ 视频已保存: {saved_video}")

                    if i < len(shots) - 1:
                        anchor_frame = self.extractor.extract_last_frame(
                            local_video)
                        # Save frame to outputs
                        saved_frame = os.path.join(
                            frames_dir, f"shot_{sid:02d}_last.png")
                        shutil.copy(anchor_frame, saved_frame)
                        _log(f"  ✓ 锚定帧已保存: {saved_frame}")
                except Exception as e:
                    _log(f"  ⚠ 锚定提取失败: {e}")
                    anchor_frame = None

            gen_method = "t2v" if (i == 0 or had_reset) else "i2v"
            final_shots.append({
                "shot_id": sid,
                "video_url": video_url,
                "local_video": os.path.join(videos_dir, f"shot_{sid:02d}.mp4")
                if local_video else None,
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

    # ══════════════════════════════════════════════
    #  Stepped execution API
    # ══════════════════════════════════════════════

    def start_session(
        self,
        user_input: str,
        style_images: list = None,
        character_images: list = None,
    ) -> PipelineSession:
        """Create a new pipeline session with run_dir.

        Call this first, then pass the session to run_planning /
        run_generation / run_composition in sequence.
        """
        from datetime import datetime

        run_dir = os.path.join(
            BASE_OUTPUT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(run_dir, exist_ok=True)

        style_images = style_images or []
        character_images = character_images or []
        refs = classify_images(style_images, character_images)

        session = PipelineSession(
            run_dir=run_dir,
            user_input=user_input,
            expanded_input=user_input,
            style_images=list(style_images),
            character_images=list(character_images),
            refs=refs,
            phase="init",
        )
        return session

    def run_planning(
        self,
        session: PipelineSession,
        progress_callback=None,
    ) -> PipelineSession:
        """Execute the Planning phase. Updates session in-place and returns it.

        Side effects:
          - session.phase → "planned"
          - Saves planning/ artifacts to disk (storyboard, context, anchors, prompts).
        """
        # ── Sync session → self for internal method compatibility ──
        self.run_dir = session.run_dir
        self._refs = session.refs

        refs = session.refs
        img_desc = describe_for_agent(refs)
        all_paths = [r.path for r in refs]

        stages = self.config.get_pipeline_stages()
        context = {}
        logs = list(session.logs)
        errors = list(session.errors)

        def _log(msg):
            logs.append(msg)
            if progress_callback:
                progress_callback("log", msg)

        _log(f"[Run] 输出目录: {session.run_dir}")
        _log(f"[Run] 参考图片:\n{img_desc}")

        # ── Idea2Video ──
        expanded_input = session.user_input
        if len(session.user_input) < IDEA2VIDEO_MIN_CHARS:
            _log("[Idea2Video] 短输入，自动展开...")
            expanded_input = self._expand_idea(session.user_input, _log)

        # ── Agent stages ──
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
                    context[agent_id] = {
                        "error": f"Unknown: {agent_id}", "skipped": True}
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
                        task=task, context=context,
                        image_paths=image_paths)
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

        # ── Anchors ──
        anchors = self._extract_anchors(context)

        # ── Extract storyboard shots ──
        storyboard = context.get("scene_planner", {})
        shots = storyboard.get("shots", [])

        # ── Write back to session ──
        session.expanded_input = expanded_input
        session.context = context
        session.anchors = anchors
        session.shots = shots
        session.logs = logs
        session.errors = errors
        session.phase = "planned"

        # ── Save planning artifacts ──
        self._save_planning_artifacts(session)

        return session

    def run_generation(
        self,
        session: PipelineSession,
        progress_callback=None,
    ) -> PipelineSession:
        """Execute the Generation phase.

        Prerequisites: session.phase == "planned".

        Side effects:
          - session.phase → "generated"
          - Saves generation/ artifacts (videos, frames, shots.json).
        """
        if session.phase != "planned":
            raise RuntimeError(
                f"run_generation requires phase='planned', "
                f"got '{session.phase}'")

        # ── Sync ──
        self.run_dir = session.run_dir
        self._refs = session.refs

        logs = list(session.logs)
        errors = list(session.errors)

        def _log(msg):
            logs.append(msg)
            if progress_callback:
                progress_callback("log", msg)

        _log("[Generation] 开始视频生成...")

        final_shots = self._generate_chain(
            session.shots, session.anchors, _log,
            max_retries=DEFAULT_MAX_RETRIES)

        # ── Write back ──
        session.final_shots = final_shots
        session.logs = logs
        session.errors = errors
        session.phase = "generated"

        # ── Save generation artifacts ──
        self._save_generation_artifacts(session)

        return session

    def run_composition(
        self,
        session: PipelineSession,
        progress_callback=None,
    ) -> PipelineSession:
        """Execute the Composition phase.

        Prerequisites: session.phase == "generated".

        Side effects:
          - session.phase → "composed"
          - Saves composition/final.mp4.
        """
        if session.phase != "generated":
            raise RuntimeError(
                f"run_composition requires phase='generated', "
                f"got '{session.phase}'")

        # ── Sync ──
        self.run_dir = session.run_dir
        self._refs = session.refs

        logs = list(session.logs)

        def _log(msg):
            logs.append(msg)
            if progress_callback:
                progress_callback("log", msg)

        comp_path = os.path.join(session.run_dir, "composition", "final.mp4")
        os.makedirs(os.path.dirname(comp_path), exist_ok=True)

        composition = self.compose(
            session.final_shots, output_path=comp_path)

        if composition.get("status") != "error":
            logs.append(
                f"[Compose] 输出: {composition.get('output_path', '')}")
        else:
            logs.append(
                f"[Compose] 失败: {composition.get('message', '')}")

        session.composition = composition
        session.logs = logs
        session.phase = "composed"

        return session

    # ══════════════════════════════════════════════
    #  Artifact persistence
    # ══════════════════════════════════════════════

    def _save_planning_artifacts(self, session: PipelineSession):
        """Save all planning-phase outputs to disk."""
        plan_dir = os.path.join(session.run_dir, "planning")
        prompts_dir = os.path.join(plan_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)

        # Full storyboard
        storyboard = session.context.get("scene_planner", {})
        with open(os.path.join(plan_dir, "storyboard.json"),
                  "w", encoding="utf-8") as f:
            json.dump(storyboard, f, ensure_ascii=False, indent=2)

        # Full agent context
        with open(os.path.join(plan_dir, "context.json"),
                  "w", encoding="utf-8") as f:
            json.dump(session.context, f, ensure_ascii=False, indent=2)

        # Anchors
        with open(os.path.join(plan_dir, "anchors.json"),
                  "w", encoding="utf-8") as f:
            json.dump(session.anchors, f, ensure_ascii=False, indent=2)

        # Per-shot unified prompts (text-level, what t2v actually receives)
        style_fragments = session.anchors.get("style_fragments", [])
        style_fragment = session.anchors.get("style_fragment", "")
        char_fragment = session.anchors.get("character_fragment", "")

        for shot in session.shots:
            sid = shot.get("shot_id", "??")
            base = shot.get("prompt", "")
            si = shot.get("style_index") or 0
            if style_fragments and 0 <= si < len(style_fragments):
                sp = style_fragments[si]
            else:
                sp = style_fragment
            unified = f"{sp}, {char_fragment}, {base}".strip(", ")

            with open(os.path.join(prompts_dir, f"shot_{sid:02d}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(unified)

        # Logs so far
        with open(os.path.join(session.run_dir, "logs.txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(session.logs))
            if session.errors:
                f.write(f"\n\n=== {len(session.errors)} ERRORS ===\n")
                for e in session.errors:
                    f.write(f"  {e}\n")

    def _save_generation_artifacts(self, session: PipelineSession):
        """Save generation-phase outputs (shots.json is the summary)."""
        import json as _json
        gen_dir = os.path.join(session.run_dir, "generation")
        os.makedirs(gen_dir, exist_ok=True)

        summary = [{
            "shot_id": s.get("shot_id", "?"),
            "method": s.get("generation_method", "?"),
            "video_url": s.get("video_url", ""),
            "local_video": s.get("local_video", ""),
            "error": s.get("error", ""),
        } for s in session.final_shots]
        with open(os.path.join(gen_dir, "shots.json"),
                  "w", encoding="utf-8") as f:
            _json.dump(summary, f, ensure_ascii=False, indent=2)

        # Append generation logs
        with open(os.path.join(session.run_dir, "logs.txt"),
                  "a", encoding="utf-8") as f:
            f.write("\n")
            # Write only the new logs since planning
            n_planning = len(
                [l for l in session.logs
                 if not l.startswith("[Generation]")
                 and not l.startswith("[Compose]")])
            for line in session.logs[n_planning:]:
                f.write(line + "\n")

    def invalidate(self):
        self.factory.invalidate()
