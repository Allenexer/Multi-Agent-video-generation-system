"""
Specialized StyleAnalyzer Agent — VLM analysis + local KMeans color extraction.

Registered with AgentFactory via @register_agent("style_analyzer").
When PipelineExecutor creates a style_analyzer agent, it gets this class
instead of the generic BaseAgent, adding automatic color palette extraction.
"""
from core.base_agent import BaseAgent
from core.agent_factory import register_agent
from tools.image_analysis import extract_color_palette


@register_agent("style_analyzer")
class StyleAnalyzerAgent(BaseAgent):
    """
    Overrides think() to append KMeans-extracted color palette to VLM result.

    VLM provides semantic style description (free text).
    KMeans provides exact hex colors (local computation, zero API cost).
    """

    def think(self, task: dict, context: dict = None,
              image_paths: list = None, use_cache: bool = True) -> dict:
        # 调用父类的think()方法，获取VLM分析结果
        result = super().think(task=task, context=context,
                               image_paths=image_paths, use_cache=use_cache)

        # 如果提供了参考图像路径，尝试提取颜色调色板并添加到结果中
        if image_paths:
            try:
                palette = extract_color_palette(image_paths[0])
                result["color_palette"] = palette
            except Exception:
                result["color_palette"] = []

        return result
