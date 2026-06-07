"""
ImageRef — typed image reference with structured labeling.

Every uploaded image gets a type + index, forming a clear contract
between the pipeline and agents. Agents reference images by label
(e.g. "风格参考0") rather than guessing from flat array order.
"""
from dataclasses import dataclass, field


@dataclass
class ImageRef:
    """A classified reference image with unique label."""
    path: str           # File path
    type: str           # "style" | "character"
    index: int          # 0-based within its type
    label: str          # e.g. "风格参考0", "角色参考0"

    def __repr__(self):
        return f"ImageRef({self.label}, {self.path})"


def classify_images(style_paths: list[str],
                    char_paths: list[str]) -> list[ImageRef]:
    """
    Build a typed, labeled image list from raw paths.

    Returns a flat list ordered: all style refs first, then character refs.
    Each image has a unique label for agent prompt reference.
    """
    refs = []
    for i, path in enumerate(style_paths or []):
        refs.append(ImageRef(
            path=path, type="style", index=i,
            label=f"风格参考{i}",
        ))
    for i, path in enumerate(char_paths or []):
        refs.append(ImageRef(
            path=path, type="character", index=i,
            label=f"角色参考{i}",
        ))
    return refs


def describe_for_agent(refs: list[ImageRef]) -> str:
    """
    Generate a text block for agent prompts describing available images.

    Example output:
      风格参考0: cyberpunk_ref.jpg (风格参考图)
      风格参考1: ink_wash_ref.jpg (风格参考图)
      角色参考0: penguin.jpg (角色参考图)
    """
    if not refs:
        return "无参考图片"

    lines = ["可用参考图片（按 label 精确引用）："]
    by_type = {}
    for r in refs:
        by_type.setdefault(r.type, []).append(r)

    type_names = {"style": "风格参考图", "character": "角色参考图"}
    for t, name in type_names.items():
        for r in by_type.get(t, []):
            lines.append(f"  {r.label}: {r.path} ({name})")
    return "\n".join(lines)


def get_by_label(refs: list[ImageRef], label: str) -> ImageRef | None:
    """Find an ImageRef by its label string."""
    for r in refs:
        if r.label == label:
            return r
    return None


def get_by_type(refs: list[ImageRef], type: str) -> list[ImageRef]:
    """Filter ImageRefs by type ('style' | 'character')."""
    return [r for r in refs if r.type == type]


def get_style_by_index(refs: list[ImageRef], index: int) -> ImageRef | None:
    """Get style ref by 0-based index."""
    styles = get_by_type(refs, "style")
    return styles[index] if 0 <= index < len(styles) else None


def get_char_by_index(refs: list[ImageRef], index: int) -> ImageRef | None:
    """Get character ref by 0-based index."""
    chars = get_by_type(refs, "character")
    return chars[index] if 0 <= index < len(chars) else None
