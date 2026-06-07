"""
CogVideoX-3 — text-to-video, image-to-video, and first-last-frame generation.

Uses zhipuai SDK (same ZHIPU_API_KEY as GLM-5V-Turbo).
Async: submit → poll → return video URL.

Three modes:
  t2v:                 prompt only
  i2v:                 single anchor image + prompt
  first-last-frame:    image_url=[first, last] + prompt (keyframe transition)
"""
import os
import time
import base64
from zhipuai import ZhipuAI

COGVIDEO_MODEL = "cogvideox-3"
COGVIDEO_POLL_INTERVAL = 5
COGVIDEO_MAX_WAIT = 300


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

    def _encode(self, path: str) -> str:
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

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

    def generate_t2v(self, prompt: str, size: str = "1280x720",
                     fps: int = 30) -> dict:
        """Text-to-video: prompt only, no image input."""
        try:
            r = self.client.videos.generations(
                model=COGVIDEO_MODEL, prompt=prompt,
                quality="speed", size=size, fps=fps)
            return self._poll(r.id)
        except Exception as e:
            return {"error": str(e)}

    def generate_i2v(self, image_path: str, prompt: str,
                     size: str = "1280x720", fps: int = 30) -> dict:
        """Image-to-video: single anchor frame + prompt."""
        try:
            r = self.client.videos.generations(
                model=COGVIDEO_MODEL,
                image_url=self._encode(image_path),
                prompt=prompt, quality="speed", size=size, fps=fps)
            return self._poll(r.id)
        except Exception as e:
            return {"error": str(e)}

    def generate_first_last_frame(self, first_frame: str, last_frame: str,
                                   prompt: str, size: str = "1280x720",
                                   fps: int = 30) -> dict:
        """First-last-frame mode: interpolate between two keyframes."""
        try:
            r = self.client.videos.generations(
                model=COGVIDEO_MODEL,
                image_url=[self._encode(first_frame),
                           self._encode(last_frame)],
                prompt=prompt, quality="quality", size=size, fps=fps)
            return self._poll(r.id)
        except Exception as e:
            return {"error": str(e)}
