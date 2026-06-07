"""Image analysis — VLM-based style analysis + local KMeans color extraction."""
import json
import base64
from config import get_vision_client
from tools.base_tool import Tool


class ImageAnalysisTool(Tool):
    """Use GLM vision model to analyze reference image style/character/composition."""

    def __init__(self):
        super().__init__(
            name="analyze_image",
            description=(
                "Analyze a reference image. Returns structured JSON with "
                "style_description, color_palette, lighting_style, line_style, "
                "composition_type, tags, and style_strength_recommendation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "analysis_focus": {
                        "type": "string",
                        "enum": ["style", "character", "composition", "all"],
                    },
                },
                "required": ["image_path"],
            },
        )
        self.client = get_vision_client()

    def execute(self, image_path: str, analysis_focus: str = "all") -> dict:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        response = self.client.chat.completions.create(
            model="glm-5v-turbo",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"Analyze the {analysis_focus} of this image. Output JSON:\n"
                        '{"style_description":"...","style_description_cn":"...",'
                        '"color_palette":["#hex"],"lighting_style":"...",'
                        '"line_style":"...","composition_type":"...",'
                        '"tags":["..."],"style_strength_recommendation":0.75}'
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(response.choices[0].message.content)


def extract_color_palette(image_path: str, n_colors: int = 5) -> list:
    """KMeans-based dominant color extraction. Zero API cost."""
    import numpy as np
    from PIL import Image
    from sklearn.cluster import KMeans

    img = Image.open(image_path).convert("RGB").resize((256, 256))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    sorted_idx = np.argsort(-counts)
    return [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in colors[sorted_idx]]
