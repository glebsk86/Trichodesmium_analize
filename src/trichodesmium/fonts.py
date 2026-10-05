"""Cyrillic report font also available in the portable Linux bundle."""
from pathlib import Path
from PIL import ImageFont


def report_font(size):
    for name in [str(Path(__file__).with_name('assets')/'DejaVuSans.ttf'),
                 'DejaVuSans.ttf','/System/Library/Fonts/Supplemental/Arial.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()
