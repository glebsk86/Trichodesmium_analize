"""Private tuning-photo regressions, explicitly NOT independent validation."""
from pathlib import Path

import numpy as np
import pytest

from trichodesmium.calibration import auto_scale
from trichodesmium.imaging import read_image, field_mask
from trichodesmium.segmentation import microscope_candidates

INPUT=Path(__file__).resolve().parents[1]/"data"/"user-six"
# Approximate points on conspicuous portions, not complete reviewed masks.
POINTS={1:(520,258),2:(650,380),3:(395,589),4:(575,706),5:(585,502),6:(608,436)}


@pytest.mark.parametrize("number",range(1,7))
def test_tuning_photo_keeps_main_hint_and_excludes_black_surround(number):
    source=INPUT/f"photo_{number}_2026-10-02_20-13-31.jpg"
    if not source.is_file():
        pytest.skip("Private tuning photographs are not in Git")
    rgb=read_image(source)
    labels,info=microscope_candidates(rgb)
    assert info["source_dimensions_match_training"]
    assert 1<=labels.max()<=8  # Regression alarm; detector has no count cap.
    assert np.mean(labels>0)<.03
    assert np.all(field_mask(rgb)[labels>0])
    x,y=POINTS[number]
    assert (labels[y-4:y+5,x-4:x+5]>0).any()
    cal=auto_scale(rgb)
    # Same objective and identical raster dimensions: approximately 3.8 px/tick.
    # This broad sanity check does not establish physical scale accuracy.
    assert cal.um_per_px==pytest.approx(.79,abs=.04)
