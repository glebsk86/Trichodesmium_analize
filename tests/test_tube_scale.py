from pathlib import Path
import copy
import numpy as np
import pytest
from trichodesmium.tube_scale import image_scale, check_scale_consistency, FINE_METHOD
from trichodesmium.imaging import read_image


def scale(value,size=(960,1280)):
    return dict(source_size_xy=list(size),source_calibration={'um_per_px':value},
                working_um_per_px_xy=None if value is None else [value,value],notice='')


def test_comparison_does_not_mix_resolutions_or_adjust_outliers():
    scales=[scale(.8),scale(.81),scale(.79),scale(.08),scale(.5,(1536,2048)),scale(None)]
    original=copy.deepcopy(scales)
    summary=check_scale_consistency(scales)
    assert scales[3]['consistency']['status']=='outlier_requires_review'
    assert scales[3]['working_um_per_px_xy']==original[3]['working_um_per_px_xy']
    assert scales[4]['consistency']['status']=='insufficient_comparison_frames'
    assert scales[5]['consistency']['status']=='scale_unavailable'
    assert len(summary['groups'])==2
    assert summary['groups'][0]['outliers']==1
    assert scales[0]['consistency']['status']=='consistent'


def test_source_to_working_scale_is_explicit():
    seed={'um_per_px':.5,'method':'auto','axes':[{'spacing_method':FINE_METHOD}]}
    result=image_scale(np.zeros((2048,1536,3),np.uint8),seed,[.625,.625])
    assert result['working_um_per_px_xy']==pytest.approx([.8,.8])
    assert result['source_calibration']['um_per_px']==.5
    assert result['source_size_xy']==[1536,2048]


@pytest.mark.parametrize('name',['04-IMG_6808.jpeg','05-IMG_6809.jpeg'])
def test_faint_ruler_windows_do_not_use_coarse_marks_as_small_ticks(name):
    path=Path('data/extra-five-2026-10-05')/name
    if not path.exists():pytest.skip('Private regression image absent')
    result=image_scale(read_image(path),{'method':'auto','um_per_px':.0789,'axes':[{'spacing_px':38.}]},[1.,1.])
    calibration=result['source_calibration']
    assert .74<calibration['um_per_px']<.84
    assert len(calibration['axes'])==2
    assert all(a['spacing_method']==FINE_METHOD for a in calibration['axes'])
    assert all(len(a['windows'])>=2 for a in calibration['axes'])
    assert result['seed_calibration']['um_per_px']==.0789
