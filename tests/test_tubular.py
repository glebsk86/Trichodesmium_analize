import cv2
import numpy as np
import pytest

from trichodesmium.tubular import TubeSettings, compare_proposal, BoundaryEvidence, fit_spine, pair_overlap, boundary_regularity


def painted(mask):
    rgb=np.full((*mask.shape,3),(195,195,180),np.uint8)
    border=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
    rgb[border]=(105,110,95)
    rgb[mask]=(165,180,105)
    return rgb


def test_patch_must_average_multiple_pixels_below_minimum_width():
    TubeSettings(min_width_px=4,patch_side_px=2).validate()
    with pytest.raises(ValueError):TubeSettings(min_width_px=4,patch_side_px=4).validate()
    with pytest.raises(ValueError):TubeSettings(patch_side_px=1).validate()


def test_round_debris_cannot_be_accepted_as_a_tube():
    mask=np.zeros((170,240),np.uint8);cv2.circle(mask,(100,80),22,1,-1)
    result=compare_proposal(painted(mask>0),mask>0,np.ones(mask.shape,bool),np.zeros(mask.shape,bool))
    assert all(v['status']=='rejected' for v in result['methods'].values())


def test_straight_tube_and_relative_hypothesis_weights():
    mask=np.zeros((150,340),np.uint8);cv2.line(mask,(30,75),(300,75),1,10)
    result=compare_proposal(painted(mask>0),mask>0,np.ones(mask.shape,bool),np.zeros(mask.shape,bool))
    for kind,method in result['methods'].items():
        assert method['status']=='accepted_candidate'
        assert len(method['fits'])==1
        fit=method['fits'][0]
        assert fit['supported_fraction']>.7
        assert fit['median_proposed_width_px']==pytest.approx(11,abs=4)
        if kind=='ensemble':
            assert len(fit['variants'])==7
            assert sum(v['relative_fit_weight'] for v in fit['variants'])==pytest.approx(1)


def test_two_crossing_axes_are_a_separate_case_and_overlap_not_measured():
    mask=np.zeros((300,340),np.uint8)
    cv2.line(mask,(25,150),(310,150),1,9)
    cv2.line(mask,(170,25),(170,270),1,9)
    result=compare_proposal(painted(mask>0),mask>0,np.ones(mask.shape,bool),np.zeros(mask.shape,bool))
    for method in result['methods'].values():
        assert method['status']=='crossing_requires_review'
        assert len(method['fits'])==2
        assert method['crossing_mask'].any()
        for fit in method['fits']:assert not (fit['supported']&method['crossing_mask']).any()


def test_real_gap_and_obstructing_ruler_remain_outside_observation_mask():
    mask=np.zeros((150,340),np.uint8);cv2.line(mask,(30,75),(300,75),1,10)
    obstruction=np.zeros(mask.shape,bool);obstruction[:,155:172]=True
    source=(mask>0)&~obstruction
    rgb=painted(mask>0);rgb[obstruction]=(80,80,80)
    result=compare_proposal(rgb,source,np.ones(mask.shape,bool),obstruction)
    assert 'gap_bridged_for_template_only' in result['axis_cases']
    for method in result['methods'].values():
        assert len(method['fits'])==1
        fit=method['fits'][0]
        assert (fit['template']&obstruction).any()
        assert not (fit['supported']&obstruction).any()


def test_square_patch_response_resists_one_pixel_colour_spike():
    mask=np.zeros((100,260),bool);mask[45:56,20:240]=True
    rgb=painted(mask);xy=np.c_[np.arange(40,220,2),np.full(90,50.)]
    evidence=BoundaryEvidence(rgb,np.ones(mask.shape,bool),TubeSettings())
    result=fit_spine(evidence,mask,xy,np.zeros(mask.shape,bool),'ensemble')
    rgb[44,120]=(255,0,255)
    noisy=fit_spine(BoundaryEvidence(rgb,np.ones(mask.shape,bool),TubeSettings()),mask,xy,np.zeros(mask.shape,bool),'ensemble')
    assert abs(result['median_proposed_width_px']-noisy['median_proposed_width_px'])<1


def test_long_bent_u_is_not_rejected_as_round_debris():
    mask=np.zeros((300,340),np.uint8)
    xy=np.r_[np.c_[np.full(55,80),np.linspace(25,190,55)],
             np.c_[170+90*np.cos(np.linspace(np.pi,0,80)),190+90*np.sin(np.linspace(np.pi,0,80))],
             np.c_[np.full(55,260),np.linspace(190,25,55)]]
    cv2.polylines(mask,[xy.astype(np.int32)],False,1,10)
    result=compare_proposal(painted(mask>0),mask>0,np.ones(mask.shape,bool),np.zeros(mask.shape,bool))
    for method in result['methods'].values():
        assert method['status']=='accepted_candidate'
        assert len(method['fits'])==1


def test_geometrically_good_but_invisible_proposal_fails_both_methods():
    mask=np.zeros((150,340),np.uint8);cv2.line(mask,(30,75),(300,75),1,10)
    rgb=np.full((*mask.shape,3),(190,185,160),np.uint8)
    result=compare_proposal(rgb,mask>0,np.ones(mask.shape,bool),np.zeros(mask.shape,bool))
    assert set(result['methods'])=={'ensemble'}
    for method in result['methods'].values():
        assert method['status']=='rejected'
        assert method['source_coverage']==0
        assert all('insufficient_bilateral_boundary_support' in f['reasons'] for f in method['fits'])


def test_parallel_overlap_is_not_reported_as_angular_crossing():
    mask=np.zeros((150,340),np.uint8);cv2.line(mask,(30,75),(300,75),1,10)
    evidence=BoundaryEvidence(painted(mask>0),np.ones(mask.shape,bool),TubeSettings())
    xy=np.c_[np.arange(35,296,2),np.full(131,75.)]
    a=fit_spine(evidence,mask>0,xy,np.zeros(mask.shape,bool),'ensemble')
    b=fit_spine(evidence,mask>0,xy+np.array([0.,1.]),np.zeros(mask.shape,bool),'ensemble')
    overlap,kind,angle=pair_overlap(a,b,TubeSettings())
    assert overlap.any()
    assert kind=='shared_axis_or_touching'
    assert angle<25


def test_touching_tips_are_not_promoted_to_resolved_crossing():
    mask=np.zeros((240,340),np.uint8)
    cv2.line(mask,(25,120),(170,120),1,10)
    cv2.line(mask,(170,120),(280,200),1,10)
    evidence=BoundaryEvidence(painted(mask>0),np.ones(mask.shape,bool),TubeSettings())
    a=np.c_[np.linspace(25,170,75),np.full(75,120.)]
    b=np.linspace([170.,120.],[280.,200.],75)
    first=fit_spine(evidence,mask>0,a,np.zeros(mask.shape,bool),'ensemble')
    second=fit_spine(evidence,mask>0,b,np.zeros(mask.shape,bool),'ensemble')
    overlap,kind,angle=pair_overlap(first,second,TubeSettings())
    assert overlap.any()
    assert angle>25
    assert kind=='endpoint_junction'


def test_straight_and_large_radius_edges_outrank_wavy_or_tight_edges():
    x=np.arange(0,300,2.);xy=np.c_[x,np.zeros(len(x))];r=np.full(len(x),5.)
    straight=boundary_regularity(xy,r,r)['boundary_regularity']
    wavy=boundary_regularity(xy,r+1.5*np.sin(x/5),r+1.5*np.sin(x/5))['boundary_regularity']
    t=np.linspace(0,np.pi/2,len(x))
    large=boundary_regularity(np.c_[100*np.cos(t),100*np.sin(t)],r,r)['boundary_regularity']
    tight=boundary_regularity(np.c_[20*np.cos(t),20*np.sin(t)],r,r)['boundary_regularity']
    assert straight>.999
    assert large>.95
    assert wavy<.8 and tight<large-.15


def test_boundary_regularity_uses_relative_width_not_absolute_pixel_radius():
    t=np.linspace(0,np.pi/2,150);xy=np.c_[100*np.cos(t),100*np.sin(t)];r=np.full(150,5.)
    one=boundary_regularity(xy,r,r)['boundary_regularity']
    two=boundary_regularity(xy*2,r*2,r*2)['boundary_regularity']
    assert two==pytest.approx(one,abs=.005)


def test_smoothness_can_be_disabled_without_changing_point_baseline():
    mask=np.zeros((150,340),np.uint8);cv2.line(mask,(30,75),(300,75),1,10)
    zero=compare_proposal(painted(mask>0),mask>0,np.ones(mask.shape,bool),np.zeros(mask.shape,bool),TubeSettings(smoothness_weight=0))
    revised=compare_proposal(painted(mask>0),mask>0,np.ones(mask.shape,bool),np.zeros(mask.shape,bool))
    assert set(zero['methods'])=={'ensemble'}
    assert zero['methods']['ensemble']['fits'][0]['smoothness_bonus']==0
    assert revised['methods']['ensemble']['fits'][0]['smoothness_bonus']>2.9
    with pytest.raises(ValueError):TubeSettings(smoothness_weight=-1).validate()


def test_caps_do_not_extend_past_a_flat_observed_end_but_keep_round_cells():
    from trichodesmium.tubular import render_tube
    xy=np.c_[np.arange(30.,301.,2),np.full(136,75.)]
    radii=np.full(len(xy),5.)
    flat=np.zeros((150,340),bool);flat[70:81,30:301]=True
    unconstrained=render_tube(flat.shape,xy,radii,radii)
    fixed=render_tube(flat.shape,xy,radii,radii,tip_support=flat)
    assert np.nonzero(unconstrained)[1].min()==25
    assert np.nonzero(fixed)[1].min()==29  # one pixel tolerance, not a full cap
    assert np.nonzero(fixed)[1].max()==301
    assert fixed[75,30:301].all()
    round_cells=np.zeros(flat.shape,np.uint8);cv2.line(round_cells,(30,75),(300,75),1,10)
    rounded=render_tube(flat.shape,xy,radii,radii,tip_support=round_cells>0)
    assert np.array_equal(rounded,unconstrained)


def test_unsupported_tip_does_not_acquire_a_circular_cap():
    from trichodesmium.tubular import render_tube
    xy=np.c_[np.arange(30.,301.,2),np.full(136,75.)]
    radii=np.full(len(xy),5.);valid=np.ones(len(xy),bool);valid[-4:]=False
    source=np.ones((150,340),bool)
    result=render_tube(source.shape,xy,radii,radii,valid=valid,tip_support=source)
    assert np.nonzero(result)[1].max()==292
