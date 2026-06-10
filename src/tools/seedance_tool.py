"""
Seedance 2.0 — video generation via Volcano Engine Ark.

Same ARK_API_KEY as Seedream. Accepts text prompt + typed reference
images/videos/audio via content[] array. Async: submit → poll → return URL.

CRITICAL: task_id is logged AND persisted to disk on every run so that
interrupted tasks can be recovered from the Volcano Engine console.
"""
import os
import json
import time
import base64
from volcenginesdkarkruntime import Ark

SEEDANCE_MODEL = "doubao-seedance-2-0-260128"
SEEDANCE_POLL_INTERVAL = 10
SEEDANCE_MAX_WAIT = 3600  # 60 min


class SeedanceTool:
    """Video generation via Seedance 2.0 (Volcano Engine Ark)."""

    def __init__(self):
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=os.environ.get("ARK_API_KEY", ""),
        )

    @staticmethod
    def _encode_image(path: str) -> str:
        """Encode a local image file to a base64 data URI."""
        with open(path, "rb") as f:
            return (
                f"data:image/png;base64,"
                f"{base64.b64encode(f.read()).decode()}"
            )

    def generate(
        self,
        prompt: str,
        reference_images: list[str] = None,
        duration: int = 5,
        run_dir: str = "",
        progress_callback=None,
    ) -> dict:
        """
        Generate video via Seedance 2.0.

        Args:
            prompt: Text prompt describing the full segment action.
            reference_images: Local file paths to reference images.
            duration: Target video duration in seconds.
            run_dir: Output directory for task_id persistence.

        Returns: {"video_url": str, "model": str, "task_id": str} or {"error": str}
        """
        content = [{"type": "text", "text": prompt or ""}]

        if reference_images:
            for path in reference_images:
                if os.path.exists(path):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(path)},
                        "role": "reference_image",
                    })

        if progress_callback:
            progress_callback(
                "log", f"[Seedance] 提交生成 (时长{duration}s, "
                f"{len(content) - 1} 张参考图)...")

        try:
            create_result = self.client.content_generation.tasks.create(
                model=SEEDANCE_MODEL,
                content=content,
                generate_audio=False,
                duration=duration,
                watermark=True,
            )
        except Exception as e:
            return {"error": str(e)}

        task_id = create_result.id
        # ── PERSIST task_id immediately ──
        self._save_task(task_id, run_dir)
        if progress_callback:
            progress_callback(
                "log",
                f"[Seedance] ⚠ task_id={task_id} — 已保存到 "
                f"{os.path.join(run_dir, 'seedance_task.json') if run_dir else '内存'} "
                f"如中断可去控制台查询: "
                f"https://console.volcengine.com/ark/region:ark+cn-beijing")

        return self._poll(task_id, run_dir, progress_callback)

    def _poll(self, task_id: str, run_dir: str = "",
              progress_callback=None) -> dict:
        elapsed = 0
        while elapsed < SEEDANCE_MAX_WAIT:
            time.sleep(SEEDANCE_POLL_INTERVAL)
            elapsed += SEEDANCE_POLL_INTERVAL
            try:
                result = self.client.content_generation.tasks.get(
                    task_id=task_id)
            except Exception as e:
                err = {"error": str(e), "task_id": task_id}
                self._save_result(task_id, run_dir, err)
                return err

            if result.status == "succeeded":
                video_url = self._extract_video_url(result)
                if not video_url:
                    err = {"error": "No video_url in response",
                           "raw": str(result)[:500], "task_id": task_id}
                    self._save_result(task_id, run_dir, err)
                    return err
                out = {"video_url": video_url, "model": SEEDANCE_MODEL,
                       "task_id": task_id}
                self._save_result(task_id, run_dir, out)
                return out

            elif result.status == "failed":
                err_msg = str(getattr(result, 'error', 'Unknown error'))
                err = {"error": err_msg, "task_id": task_id}
                self._save_result(task_id, run_dir, err)
                return err

            if progress_callback:
                progress_callback(
                    "log",
                    f"[Seedance] {result.status} ({elapsed}s)...")

        err = {"error": "Timeout", "task_id": task_id}
        self._save_result(task_id, run_dir, err)
        return err

    @staticmethod
    def _extract_video_url(result) -> str:
        """Try multiple possible response structures for video_url."""
        # Path 1: result.content.video_url (dict, from task query)
        if hasattr(result, 'content') and hasattr(result.content, 'video_url'):
            return result.content.video_url
        # Path 2: result.content[].video_url (list, from stream event)
        if hasattr(result, 'content') and result.content:
            try:
                for item in result.content:
                    vu = getattr(item, 'video_url', None)
                    if vu:
                        return vu
            except TypeError:
                pass
        # Path 2: result.generated_videos[].video
        if hasattr(result, 'generated_videos') and result.generated_videos:
            vu = getattr(result.generated_videos[0], 'video', None)
            if vu:
                return vu
        # Path 3: result.output.video_url
        if hasattr(result, 'output'):
            vu = getattr(result.output, 'video_url', '')
            if vu:
                return vu
        # Path 4: result.video_url
        vu = getattr(result, 'video_url', '')
        if vu:
            return vu
        # Path 5: regex search in raw string
        raw = str(result)
        import re
        m = re.search(r'https?://[^\s\"\'<>]+\.mp4', raw)
        if m:
            return m.group(0)
        return ""

    @staticmethod
    def _save_task(task_id: str, run_dir: str):
        """Persist task_id immediately so orphaned tasks are recoverable."""
        if not run_dir:
            return
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "seedance_task.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"task_id": task_id, "status": "submitted",
                        "console_url": "https://console.volcengine.com/ark/"
                        "region:ark+cn-beijing"}, f, ensure_ascii=False)

    @staticmethod
    def _save_result(task_id: str, run_dir: str, result: dict):
        """Update persisted task file with final result."""
        if not run_dir:
            return
        path = os.path.join(run_dir, "seedance_task.json")
        data = {"task_id": task_id}
        data.update(result)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
