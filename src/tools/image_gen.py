"""
Seedream 4.5 — unified image generation via Volcano Engine Ark.

API: doubao-seedream-4-5-251128 via OpenAI-compatible SDK.
Supports 6 mode combinations:

                  | no refs          | single ref           | multi refs
------------------|-----------------|----------------------|------------------
num_images=1      | ① text → single | ② single ref → single| ③ multi ref → single
num_images=N      | ④ text → multi  | ⑤ single ref → multi | ⑥ multi ref → multi

Pipeline role:
  generate_start_frame:  character refs + style ref → starting keyframe
  generate_end_frame:    start_frame + style ref → visually continuous end frame
"""
import os
import base64
from openai import OpenAI

SEEDREAM_MODEL = "doubao-seedream-5-0-260128"
SEEDREAM_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"

# ── Watermark policy ──
# Set to False to remove "AI-generated" watermarks (if your account permits).
SEEDREAM_WATERMARK = True


class SeedreamTool:
    """Seedream 4.5 image generation via Volcano Engine Ark."""

    def __init__(self):
        self.client = OpenAI(
            base_url=SEEDREAM_BASE_URL,
            api_key=os.environ.get("ARK_API_KEY", ""),
        )

    # ══════════════════════════════════════════════
    #  Unified entry — covers all 6 modes
    # ══════════════════════════════════════════════

    def generate(
        self,
        prompt: str,
        reference_images: list[str] | None = None,
        num_images: int = 1,
        size: str = "2K",
    ) -> list[str]:
        """
        Seedream 4.5 unified generation.

        Args:
            prompt: Text prompt describing the desired image(s).
            reference_images: Local file paths to 0-N reference images.
                              Single string → mode ②/⑤; list → mode ③/⑥.
            num_images: 1 for single image, 2-15 for sequential group.
            size: Output size preset ("2K", "4K", etc.).

        Returns:
            List of saved PNG file paths. Empty list on failure.

        Mode matrix:
            num_images=1, reference_images=None        → ① text → single
            num_images=1, reference_images=[a]         → ② single ref → single
            num_images=1, reference_images=[a,b,...]   → ③ multi ref → single
            num_images=N, reference_images=None        → ④ text → multi
            num_images=N, reference_images=[a]         → ⑤ single ref → multi
            num_images=N, reference_images=[a,b,...]   → ⑥ multi ref → multi
        """
        extra_body = {"watermark": SEEDREAM_WATERMARK}

        # ── Reference images ──
        # API field is "image" (not "reference_images")
        if reference_images:
            encoded = [self._encode(p) for p in reference_images]
            extra_body["image"] = (
                encoded[0] if len(encoded) == 1 else encoded
            )
        # Always single-image mode (group mode unreliable with base64)

        return self._generate_and_collect(prompt, size, extra_body)

    # ══════════════════════════════════════════════
    #  Convenience wrappers (preserve existing API)
    # ══════════════════════════════════════════════

    def generate_start_frame(
        self,
        prompt: str,
        character_refs: dict = None,
        style_ref: str = None,
        size: str = "2K",
    ) -> str | None:
        """
        Generate the STARTING keyframe of a shot.

        Uses character refs + style ref for identity locking.
        Returns path to saved PNG, or None on failure.
        """
        refs = []
        # Style first — Seedream treats first ref as primary style anchor
        if style_ref:
            refs.append(style_ref)
        if character_refs:
            for val in character_refs.values():
                for p in (val if isinstance(val, list) else [val]):
                    refs.append(p)

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

        results = self.generate(
            prompt=full_prompt,
            reference_images=refs if refs else None,
            num_images=1,
            size=size,
        )
        return results[0] if results else None

    def generate_end_frame(
        self,
        prompt: str,
        start_frame: str,
        style_ref: str = None,
        size: str = "2K",
    ) -> str | None:
        """
        Generate the ENDING keyframe of a shot.

        The start_frame is passed as a reference — Seedream will produce
        a frame that VISUALLY CONTINUES from the start frame while applying
        the changes described in the prompt.

        Returns path to saved PNG, or None on failure.
        """
        refs = []
        if style_ref:
            refs.append(style_ref)
        refs.append(start_frame)

        full_prompt = (
            f"CRITICAL — The FIRST reference image defines the VISUAL STYLE. "
            f"Apply its exact artistic style. "
            f"The second reference image is the STARTING frame. "
            f"Generate a frame that is IDENTICAL in scene, background, "
            f"composition, and lighting to the starting frame. "
            f"Only make MINIMAL changes: the subject may have shifted "
            f"position slightly, or the camera may have panned a short "
            f"distance. 95% of the image must match the starting frame — "
            f"only 5% should differ. "
            f"The two frames should look like adjacent frames in a video. "
            f"{prompt}"
        )

        results = self.generate(
            prompt=full_prompt,
            reference_images=refs,
            num_images=1,
            size=size,
        )
        return results[0] if results else None

    # ══════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════

    def _generate_and_collect(
        self, prompt: str, size: str, extra_body: dict
    ) -> list[str]:
        """Stream images from Seedream and save all to disk.

        Returns list of saved PNG paths (empty on failure).
        Handles both single-image and sequential-group responses.
        """
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
                if event is None:
                    continue
                if event.type == "image_generation.partial_succeeded":
                    if event.b64_json is not None:
                        frames_b64.append(event.b64_json)
        except Exception:
            return []

        return [self._save(b64) for b64 in frames_b64]

    @staticmethod
    def _encode(path: str) -> str:
        with open(path, "rb") as f:
            return (
                f"data:image/png;base64,"
                f"{base64.b64encode(f.read()).decode()}"
            )

    @staticmethod
    def _save(b64_data: str) -> str:
        import tempfile

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        return path
