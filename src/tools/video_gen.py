"""
CogVideoX-3 — text-to-video, image-to-video, and first-last-frame generation.

Uses zhipuai SDK (same ZHIPU_API_KEY as GLM-5V-Turbo).
Async: submit → poll → return video URL.

Three modes:
  t2v:                 prompt only (upgraded to i2v when style_image provided)
  i2v:                 single anchor image + prompt
                       (upgraded to first-last-frame when style_image provided)
  first-last-frame:    image_url=[first, last] + prompt (keyframe transition)

Style injection: when a style reference image is available, it is passed
directly as image_url to CogVideoX, giving the model pixel-level visual
guidance instead of relying solely on text descriptions.
"""
import os
import time
import base64
from zhipuai import ZhipuAI

COGVIDEO_MODEL = "cogvideox-3"
COGVIDEO_POLL_INTERVAL = 10    # quality mode needs longer, poll less often
COGVIDEO_MAX_WAIT = 1800        # 1080p + 60fps + quality + audio, up to 30 min

# ── Upgraded defaults (was: 1280x720 / 30fps / speed / no audio) ──
COGVIDEO_DEFAULT_SIZE = "1920x1080"
COGVIDEO_DEFAULT_FPS = 60
COGVIDEO_DEFAULT_QUALITY = "quality"


class CogVideoXTool:
    """
    Video generation via Zhipu CogVideoX-3.

    Pipeline role:
      Shot 1 (no keyframes):        t2v
      Shot 2..N (no keyframes):     i2v (anchored to previous shot's last frame)
      Any shot (with keyframes):    first-last-frame mode
    """

    def __init__(self):
        self.client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY", ""))

    @staticmethod
    def _mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "image/png")

    def _encode(self, path: str) -> str:
        with open(path, "rb") as f:
            return f"data:{self._mime(path)};base64,{base64.b64encode(f.read()).decode()}"

    def _poll(self, task_id: str) -> dict:
        elapsed = 0
        while elapsed < COGVIDEO_MAX_WAIT:
            time.sleep(COGVIDEO_POLL_INTERVAL)
            elapsed += COGVIDEO_POLL_INTERVAL
            result = self.client.videos.retrieve_videos_result(id=task_id)
            if result.task_status == "SUCCESS":
                videos = result.video_result
                return {"video_url": videos[0].url if videos else "",
                        "model": COGVIDEO_MODEL}
            elif result.task_status == "FAILED":
                return {"error": "Generation failed", "task_id": task_id}
        return {"error": "Timeout", "task_id": task_id}

    def generate_t2v(self, prompt: str, size: str = COGVIDEO_DEFAULT_SIZE,
                     fps: int = COGVIDEO_DEFAULT_FPS,
                     style_image: str = None) -> dict:
        """Text-to-video. When style_image provided, upgrades to i2v
        so CogVideoX receives pixel-level style guidance."""
        # ── Style injection: t2v → i2v with style ref as visual anchor ──
        if style_image:
            return self.generate_i2v(
                image_path=style_image, prompt=prompt,
                size=size, fps=fps)
        try:
            r = self.client.videos.generations(
                model=COGVIDEO_MODEL, prompt=prompt,
                quality=COGVIDEO_DEFAULT_QUALITY,
                size=size, fps=fps,
                with_audio=True)
            return self._poll(r.id)
        except Exception as e:
            return {"error": str(e)}

    def generate_i2v(self, image_path: str, prompt: str,
                     size: str = COGVIDEO_DEFAULT_SIZE,
                     fps: int = COGVIDEO_DEFAULT_FPS,
                     style_image: str = None) -> dict:
        """Image-to-video. When style_image provided, upgrades to
        first-last-frame mode with [anchor, style_ref] for style injection."""
        # ── Style injection: i2v → first-last-frame with style ref ──
        if style_image:
            return self.generate_first_last_frame(
                first_frame=image_path, last_frame=style_image,
                prompt=prompt, size=size, fps=fps)
        try:
            r = self.client.videos.generations(
                model=COGVIDEO_MODEL,
                image_url=self._encode(image_path),
                prompt=prompt,
                quality=COGVIDEO_DEFAULT_QUALITY,
                size=size, fps=fps,
                with_audio=True)
            return self._poll(r.id)
        except Exception as e:
            return {"error": str(e)}

    def generate_first_last_frame(self, first_frame: str, last_frame: str,
                                   prompt: str,
                                   size: str = COGVIDEO_DEFAULT_SIZE,
                                   fps: int = COGVIDEO_DEFAULT_FPS) -> dict:
        """First-last-frame mode: interpolate between two keyframes."""
        try:
            r = self.client.videos.generations(
                model=COGVIDEO_MODEL,
                image_url=[self._encode(first_frame),
                           self._encode(last_frame)],
                prompt=prompt,
                quality=COGVIDEO_DEFAULT_QUALITY,
                size=size, fps=fps,
                with_audio=True)
            return self._poll(r.id)
        except Exception as e:
            return {"error": str(e)}
