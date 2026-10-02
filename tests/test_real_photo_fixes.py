"""Failure-oriented checks with known geometry; originals remain private data."""
import cv2
import numpy as np
from PIL import Image
import pytest

from trichodesmium.calibration import analyse_axis, auto_scale
from trichodesmium.imaging import candidates, field_mask, read_labels
from trichodesmium.segmentation import load_profile, microscope_candidates
from trichodesmium.geometry import measure


def test_tinted_microscope_field_is_not_one_coloured_object():
    rgb=np.zeros((400,400,3),np.uint8)
    cv2.circle(rgb,(200,200),180,(190,178,147),-1)
    # Simulate JPEG noise in the black surround.
    rng=np.random.default_rng(4)
    rgb[rgb[...,0]==0]=rng.integers(0,18,(np.sum(rgb[...,0]==0),3),dtype=np.uint8)
    assert not candidates(rgb).any()
    labels,_=microscope_candidates(rgb)
    assert not labels.any()


def test_field_mask_preserves_dark_internal_filament_not_black_surround():
    rgb=np.zeros((300,300,3),np.uint8)
    cv2.circle(rgb,(150,150),120,(180,170,150),-1)
    cv2.line(rgb,(90,90),(210,210),(10,10,10),8)
    field=field_mask(rgb)
    assert field[150,150]
    assert not field[3,3]


@pytest.mark.parametrize("spacing",[3.8,12.])
def test_blurred_fine_ticks_and_coarse_ticks_keep_true_interval(spacing):
    # Render at 4x so the known interval is not rounded to integral pixels.
    image=np.full((320,2400,3),(190,181,166),np.uint8)
    cv2.line(image,(60,160),(2340,160),(125,120,112),4)
    for x in np.arange(64,2340,spacing*4):
        cv2.line(image,(round(x),136),(round(x),184),(128,124,117),3)
    image=cv2.GaussianBlur(image,(0,0),1.8)
    image=cv2.resize(image,(600,80),interpolation=cv2.INTER_AREA)
    result=analyse_axis(image,np.array([15.,40.]),np.array([585.,40.]))
    assert result is not None
    assert result["spacing_px"]==pytest.approx(spacing,rel=.04)


def test_browser_export_keeps_disconnected_parts_of_one_id(tmp_path):
    labels=np.zeros((24,30),np.uint8)
    labels[2:6,3:8]=1
    labels[15:20,18:23]=1
    rgba=np.repeat(labels[...,None],4,axis=2)
    rgba[...,3]=255
    path=tmp_path/"edited.png"
    Image.fromarray(rgba).save(path)
    result=read_labels(path,(24,30,3),"instances")
    np.testing.assert_array_equal(result,labels)
    assert read_labels(path,(24,30,3),"binary").max()==2
    rgba[3,4,1]=2
    Image.fromarray(rgba).save(path)
    with pytest.raises(ValueError,match="identical grayscale"):
        read_labels(path,(24,30,3),"instances")


def test_shipped_profile_integrity_and_training_disclosure():
    model,metadata=load_profile()
    assert model.isTrained()
    assert len(metadata["training_images"])==6
    assert "No independent validation" in metadata["training_notice"]


def test_short_edge_spur_is_flagged_but_real_crossing_stays_unresolved():
    mask=np.zeros((180,240),np.uint8)
    cv2.line(mask,(30,90),(210,90),1,12)
    # A short thin twig is an edge defect, not a second long trichome.
    cv2.line(mask,(120,90),(120,79),1,3)
    rgb=np.full((180,240,3),180,np.uint8)
    result=measure(mask.astype(bool),rgb,np.zeros_like(mask,bool),prune_spurs=True)
    assert result["length_px"]==pytest.approx(192,abs=2)
    assert "short_skeleton_spurs_ignored" in result["flags"]
    cv2.line(mask,(120,20),(120,160),1,12)
    result=measure(mask.astype(bool),rgb,np.zeros_like(mask,bool),prune_spurs=True)
    assert result["length_px"] is None
    assert "branched_or_closed_mask" in result["flags"]
