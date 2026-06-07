"""
tools/ — External API integrations and video processing.

Modules:
  base_tool.py       — Tool base class
  image_analysis.py  — VLM image analysis + KMeans color extraction
  image_gen.py       — Seedream 4.5 character-consistent keyframe generation
  video_gen.py       — CogVideoX-3 text-to-video / image-to-video / first-last-frame
  video_compose.py   — Frame extraction (OpenCV) + video concatenation (moviepy)
"""
from tools.base_tool import Tool
from tools.image_analysis import ImageAnalysisTool, extract_color_palette
from tools.image_gen import SeedreamTool
from tools.video_gen import CogVideoXTool
from tools.video_compose import KeyframeExtractor, VideoConcatTool

__all__ = [
    "Tool",
    "ImageAnalysisTool", "extract_color_palette",
    "SeedreamTool",
    "CogVideoXTool",
    "KeyframeExtractor", "VideoConcatTool",
]
