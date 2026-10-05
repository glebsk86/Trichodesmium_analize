import numpy as np
import pytest

from trichodesmium.tube_guides import measurement_guides, guide_label, with_header


def straight_fit(step=1):
    x=np.arange(30,301,step,dtype=float);xy=np.c_[x,np.full(len(x),75.)]
    mask=np.zeros((150,340),bool);mask[70:81,30:301]=True
    return {'axis_xy':xy,'supported':mask,'left_radius_px':np.full(len(x),5.),
            'right_radius_px':np.full(len(x),5.),'reasons':[]}


def test_drawn_lines_are_exactly_the_samples_used_in_width_and_length():
    fit=straight_fit();result=measurement_guides(fit)
    assert result['visible_axis_length_px']==pytest.approx(270.)
    assert result['mean_fitted_width_px']==pytest.approx(10.)
    assert len(result['width_lines_xy'])==12
    for a,b in np.array(result['width_lines_xy']):
        assert a[0]==b[0]
        assert np.linalg.norm(b-a)==pytest.approx(10.)
    assert sum(result['length_segments_px'])==pytest.approx(result['visible_axis_length_px'])
    assert result['full_chain_length_px'] is None
    assert result['length_um'] is None and result['width_um'] is None


def test_unsupported_junction_is_not_in_length_or_width_samples():
    fit=straight_fit();fit['supported'][:,140:160]=False
    result=measurement_guides(fit)
    assert result['visible_axis_length_px']==pytest.approx(249.)
    assert result['visible_axis_length_px']<270
    assert all(not 140<=a[0]<160 for a,b in result['width_lines_xy'])
    assert all(not 140<=p[0]<160 for segment in result['length_segments_xy'] for p in segment)


def test_one_pixel_gap_between_axis_samples_is_not_bridged():
    fit=straight_fit(step=2);fit['supported'][:,151]=False
    result=measurement_guides(fit)
    assert result['visible_axis_length_px']==pytest.approx(268.)


def test_rejected_fit_does_not_receive_measurement_values():
    fit=straight_fit();fit['reasons']=['insufficient_bilateral_boundary_support']
    result=measurement_guides(fit)
    assert result['visible_axis_length_px'] is None
    assert result['mean_fitted_width_px'] is None
    assert result['width_lines_xy']==[] and result['length_segments_xy']==[]


def test_narrow_object_caption_is_wrapped_above_the_photo():
    label=guide_label('1.1',measurement_guides(straight_fit()))
    image,header=with_header(np.full((100,300,3),190,np.uint8),[label])
    assert image.size==(300,100+header)
    assert header>60


def test_physical_dimensions_follow_exact_drawn_vectors_and_raster_scale():
    scale={'working_um_per_px_xy':[.5/.625,.5/.625]}
    guides=measurement_guides(straight_fit(),calibration=scale)
    assert guides['length_um']==pytest.approx(216.)
    assert guides['width_um']==pytest.approx(8.)
    assert sum(guides['length_segments_um'])==pytest.approx(guides['length_um'])
    assert np.mean(guides['width_samples_um'])==pytest.approx(guides['width_um'])
    assert 'мкм' in guide_label('1.1',guides)
    # X and Y factors remain distinct after an anisotropic image resize.
    guides=measurement_guides(straight_fit(),calibration={'working_um_per_px_xy':[.8,.6]})
    assert guides['length_um']==pytest.approx(216.)
    assert guides['width_um']==pytest.approx(6.)


def test_invalid_physical_scale_is_rejected():
    with pytest.raises(ValueError,match='positive finite'):
        measurement_guides(straight_fit(),calibration={'working_um_per_px_xy':[0,.8]})
