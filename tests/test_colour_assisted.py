"""Geometry checks and private capture regressions, not biological validation."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from trichodesmium.geometry import measure
from trichodesmium.imaging import field_mask, read_image
from trichodesmium.segmentation import (
    COLOUR_PROFILE, associate_supported_pixels, colour_assisted_candidates, colour_evidence,
)


def test_bilateral_colour_with_edges_survives_illumination_gradient():
    rgb = np.empty((160, 320, 3), np.uint8)
    gradient = np.linspace(-12, 12, 320)
    for channel, base in enumerate((195, 195, 182)):
        rgb[..., channel] = base+gradient
    rgb[69:85, 35:285] = (110, 110, 100)  # Known paired optical borders.
    rgb[71:83, 35:285] = (172, 180, 112)
    rgb[118:122, 35:285] = (90, 90, 90)  # Achromatic ruler-like line.
    contrast, paired = colour_evidence(rgb)
    assert np.median(contrast[76, 60:260]) > 6
    assert np.median(paired[76, 60:260]) > 6
    assert np.max(paired[119:121, 60:260]) < 6
    assert np.max(contrast[30:40, 60:260]) < 1.5


def test_nearby_supported_fragments_share_id_but_gap_is_not_measured():
    support = np.zeros((130, 380), bool)
    support[50:59, 25:140] = True
    support[50:59, 142:270] = True  # Two background columns are genuinely empty.
    support[100:109, 25:270] = True  # Unseeded elongated background structure.
    seeds = np.zeros_like(support)
    seeds[50:59, 50:100] = True
    labels, _ = associate_supported_pixels(support, seeds)
    assert labels[54, 80] == labels[54, 220] == 1
    assert not labels[:, 140:142].any()
    assert not labels[100:109].any()
    assert np.all(support[labels > 0])
    rgb = np.full((*support.shape, 3), 180, np.uint8)
    metrics = measure(labels == 1, rgb, np.zeros_like(support))
    assert metrics["length_px"] is None
    assert "disconnected_instance_mask" in metrics["flags"]


def test_tiny_seed_cannot_claim_a_large_network():
    support = np.zeros((100, 500), bool)
    support[30:40, 20:480] = True
    seeds = np.zeros_like(support)
    seeds[30:40, 20:24] = True
    labels, _ = associate_supported_pixels(support, seeds)
    assert not labels.any()


ROOT = Path(__file__).resolve().parents[1]
# Visually located hints on the principal filament, not expert outer masks.
HINTS = {
    1: [(470, 175), (670, 490), (920, 880)],
    2: [(700, 380), (270, 490), (390, 730)],
    3: [(80, 725), (395, 589), (690, 465)],
    4: [(588, 240), (575, 706), (459, 962)],
    5: [(170, 765), (395, 720), (585, 502)],
    6: [(365, 538), (608, 436), (920, 309)],
    12: [(465, 70), (520, 260), (565, 400)],
    13: [(330, 335), (550, 457), (790, 541)],
    23: [(620, 395), (470, 722), (345, 1225)],
}


@pytest.mark.parametrize("number", list(HINTS))
def test_private_capture_keeps_multiple_principal_filament_hints(number):
    folder = ROOT/"data"/("user-six" if number <= 6 else "check-2026-10-05")
    source = folder/f"photo_{number}_2026-10-02_20-13-31.jpg"
    if not source.is_file():
        pytest.skip("Private capture photographs are excluded from Git")
    rgb = read_image(source)
    labels, info = colour_assisted_candidates(rgb)
    assert info["profile"] == COLOUR_PROFILE
    assert info["forest_retrained"] is False
    assert np.all(field_mask(rgb)[labels > 0])
    assert np.mean(labels > 0) < .05
    identifiers = []
    for x, y in HINTS[number]:
        patch = labels[max(0, y-8):y+9, max(0, x-8):x+9]
        ids, sizes = np.unique(patch[patch > 0], return_counts=True)
        assert len(ids), f"Missed approximate principal-filament hint {(x, y)}"
        identifiers.append(ids[np.argmax(sizes)])
    assert len(set(identifiers)) == 1
