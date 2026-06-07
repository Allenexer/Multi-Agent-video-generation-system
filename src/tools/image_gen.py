"""
Seedream 4.5 — character-consistent keyframe generation via Volcano Engine Ark.

Two modes:
  generate_start:  character refs + style ref + prompt → starting keyframe
  generate_end:    start_frame + prompt + style ref → ending keyframe
                   (visually anchored to start_frame for continuity)
"""
import os
import base64
from openai import OpenAI

SEEDREAM_MODEL = "doubao-seedream-4-5-251128"
SEEDREAM_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class SeedreamTool:

    def __init__(self):
        self.client = OpenAI(
            base_url=SEEDREAM_BASE_URL,
            api_key=os.environ.get("ARK_API_KEY", ""),
        )

    def generate_start_frame(
        self,
        prompt: str,
        character_refs: dict = None,
        style_ref: str = None,
        size: str = "2K",
    ) -> str | None:
        """
        Generate the STARTING frame of a shot.

        Uses character refs + style ref for identity locking.
        Returns path to saved PNG, or None on failure.
        """
        reference_urls = []
        if style_ref:
            reference_urls.append(self._encode(style_ref))
        if character_refs:
            for val in character_refs.values():
                for p in (val if isinstance(val, list) else [val]):
                    reference_urls.append(self._encode(p))

        full_prompt = (
            f"CRITICAL — The FIRST reference image defines the VISUAL STYLE. "
            f"Apply its exact artistic style to everything: color palette, "
            f"brush technique, lighting mood, line quality, rendering method. "
            f"The generated image must look like it was painted by the same "
            f"artist as the style reference. "
            f"Scene: {prompt} "
            f"Additional reference images show the character(s) — match their "
            f"exact appearance, then render them in the style's aesthetic."
        )

        extra_body = {
            "watermark": True,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": 1},
        }
        if reference_urls:
            extra_body["image"] = reference_urls

        return self._generate_and_save(full_prompt, size, extra_body)

    def generate_end_frame(
        self,
        prompt: str,
        start_frame: str,
        style_ref: str = None,
        size: str = "2K",
    ) -> str | None:
        """
        Generate the ENDING frame of a shot.

        The start_frame is passed as a reference — Seedream will generate
        a frame that VISUALLY CONTINUES from the start frame while applying
        the changes described in the prompt.

        Returns path to saved PNG, or None on failure.
        """
        # Style FIRST so it anchors the visual aesthetic
        reference_urls = []
        if style_ref:
            reference_urls.append(self._encode(style_ref))
        reference_urls.append(self._encode(start_frame))

        full_prompt = (
            f"CRITICAL — The FIRST reference image defines the VISUAL STYLE. "
            f"Apply its exact artistic style. "
            f"The second reference image is the STARTING frame. "
            f"Generate a frame that is IDENTICAL in scene, background, composition, "
            f"and lighting to the starting frame. Only make MINIMAL changes: "
            f"the subject may have shifted position slightly, or the camera "
            f"may have panned a short distance. 95% of the image must match "
            f"the starting frame — only 5% should differ. "
            f"The two frames should look like adjacent frames in a video. "
            f"{prompt}"
        )

        extra_body = {
            "watermark": True,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": 1},
        }
        extra_body["image"] = reference_urls

        return self._generate_and_save(full_prompt, size, extra_body)

    def _generate_and_save(self, prompt: str, size: str,
                           extra_body: dict) -> str | None:
        """Stream a single image from Seedream and save to disk."""
        frames_b64 = []
        try:
            response = self.client.images.generate(
                model=SEEDREAM_MODEL,
                prompt=prompt,
                size=size,
                response_format="b64_json",
                stream=True,
                extra_body=extra_body,
            )
            for event in response:
                if event and event.type == "image_generation.partial_succeeded":
                    if event.b64_json:
                        frames_b64.append(event.b64_json)
        except Exception:
            return None

        if not frames_b64:
            return None
        return self._save(frames_b64[0])

    @staticmethod
    def _encode(path: str) -> str:
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    @staticmethod
    def _save(b64_data: str) -> str:
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return path
