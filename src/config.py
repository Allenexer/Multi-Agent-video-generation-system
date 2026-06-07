"""
Backward-compatible re-export of core.config_manager.
Existing code importing from 'config' continues to work.
"""
from core.config_manager import ConfigManager

# Legacy convenience functions — still work but now delegate to ConfigManager
def get_text_client():
    cfg = ConfigManager.get_instance()
    return cfg.get_client_for_agent("director")  # director uses text provider

def get_vision_client():
    cfg = ConfigManager.get_instance()
    return cfg.get_client_for_agent("style_analyzer")  # style_analyzer uses vision provider
