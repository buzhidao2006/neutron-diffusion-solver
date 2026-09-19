"""Shared Matplotlib settings for readable solver charts."""

from matplotlib import font_manager, rcParams


_PREFERRED_CHINESE_FONTS = (
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "Microsoft JhengHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "PingFang SC",
    "Heiti SC",
    "WenQuanYi Zen Hei",
)


def configure_matplotlib_fonts():
    """Choose an installed CJK font when available and keep minus signs readable."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    available = [name for name in _PREFERRED_CHINESE_FONTS if name in installed]
    rcParams["font.sans-serif"] = [*available, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    return available[0] if available else None
