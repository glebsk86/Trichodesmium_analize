"""Known calibration geometry; added for the native B transition."""
import numpy as np
from trichodesmium.physical_settings import settings_for_photo


def test_radius_and_square_follow_physical_scale():
    options=dict(min_radius_um=1.5,max_radius_um=12.,patch_side_um=4.8)
    a,ra=settings_for_photo({'working_um_per_px_xy':[.8,.8]},[960,1280],options)
    b,rb=settings_for_photo({'working_um_per_px_xy':[.25,.25]},[3024,4032],options)
    assert a.patch_side_px==6 and b.patch_side_px==20
    assert np.allclose([a.min_width_px*.8,a.max_width_px*.8],[3,24])
    assert np.allclose([b.min_width_px*.25,b.max_width_px*.25],[3,24])
    assert ra['origin']==rb['origin']=='physical_um'


def test_unavailable_scale_never_applies_micrometres():
    settings,record=settings_for_photo({'working_um_per_px_xy':None},[3024,4032],{})
    assert settings.patch_side_px==19
    assert record['effective_radius_bounds_um'] is None
    assert record['patch_side_um'] is None
