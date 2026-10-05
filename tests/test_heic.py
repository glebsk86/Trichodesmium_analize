"""Real HEVC encoding/decoding, orientation and reports (no renamed JPEGs)."""
import argparse
import json
import sys

import cv2
import numpy as np
from PIL import Image,ImageOps,ImageCms
import pillow_heif
import pytest

from trichodesmium.imaging import enable_heif,read_image,list_images
from trichodesmium.cli import main
from trichodesmium.tube_compare import run


def colours():
    rgb=np.zeros((80,120,3),np.uint8)
    rgb[:40,:60]=(200,20,10);rgb[:40,60:]=(10,190,30)
    rgb[40:,:60]=(20,30,190);rgb[40:,60:]=(210,200,40)
    return rgb


def save_heic(path,rgb,orientation=1,**kwargs):
    enable_heif()
    im=Image.fromarray(rgb);im.getexif()[274]=orientation
    im.save(path,format='HEIF',quality=-1,chroma=444,matrix_coefficients=0,**kwargs)


@pytest.mark.parametrize('orientation',[1,2,3,4,5,6,7,8])
def test_full_resolution_colour_and_container_rotation_once(tmp_path,orientation):
    rgb=colours();source=tmp_path/'iphone.HEIC'
    save_heic(source,rgb,orientation)
    expected=Image.fromarray(rgb);expected.getexif()[274]=orientation
    expected=np.asarray(ImageOps.exif_transpose(expected))
    got,meta=read_image(source,return_metadata=True)
    assert got.shape==expected.shape
    assert got.dtype==np.uint8
    np.testing.assert_allclose(got,expected,atol=2)
    assert meta['format']=='HEIF' and meta['decoded_mode']=='RGB8'
    assert meta['oriented_size_px']==[expected.shape[1],expected.shape[0]]
    assert list_images(tmp_path)==[source]


def test_primary_image_is_selected_instead_of_first_container_image(tmp_path):
    enable_heif();source=tmp_path/'frames.heif'
    Image.new('RGB',(80,60),'red').save(source,format='HEIF',save_all=True,
        append_images=[Image.new('RGB',(120,90),'blue')],primary_index=1,quality=-1)
    got,meta=read_image(source,return_metadata=True)
    assert got.shape==(90,120,3)
    assert got[20,20,2]>240 and got[20,20,0]<5
    assert meta['frame_count']==2 and meta['selected_frame']==1
    assert 'only_primary_HEIF_image_processed' in meta['warnings']


def test_10bit_heic_is_explicitly_decoded_to_rgb8(tmp_path):
    source=tmp_path/'ten-bit.heic';rgb=colours().astype(np.uint16)*257
    pillow_heif.from_bytes('RGB;16',(120,80),rgb.tobytes()).save(source,quality=-1)
    got,meta=read_image(source,return_metadata=True)
    assert got.shape==(80,120,3) and got.dtype==np.uint8
    assert meta['source_bit_depth']==10
    assert 'high_bit_depth_decoded_to_RGB8' in meta['warnings']
    assert got.max()>180 and got.mean()>40


def test_embedded_icc_is_converted_to_srgb_and_invalid_profile_fails(tmp_path):
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    source=tmp_path/'icc.heic';save_heic(source,colours(),icc_profile=profile)
    rgb,meta=read_image(source,return_metadata=True)
    np.testing.assert_allclose(rgb,colours(),atol=2)
    assert meta['colour_conversion']=='embedded ICC to sRGB'
    source=tmp_path/'bad-icc.heic';save_heic(source,colours(),icc_profile=b'invalid')
    with pytest.raises((OSError,ValueError)):read_image(source)


def test_missing_decoder_explains_install_and_other_formats_work(tmp_path,monkeypatch):
    source=tmp_path/'photo.heic';save_heic(source,colours())
    png=tmp_path/'photo.png';Image.fromarray(colours()).save(png)
    enable_heif.cache_clear()
    try:
        monkeypatch.setitem(sys.modules,'pillow_heif',None)
        with pytest.raises(ValueError,match='pip install'):read_image(source)
        assert np.array_equal(read_image(png),colours())
    finally:
        enable_heif.cache_clear()


def test_heic_regular_and_b_reports_preserve_source_and_use_png_preview(tmp_path):
    photos=tmp_path/'photos';photos.mkdir();seeds=tmp_path/'seeds'
    mask=np.zeros((180,340),np.uint8);cv2.line(mask,(30,90),(300,90),1,10)
    rgb=np.full((*mask.shape,3),(195,195,180),np.uint8);rgb[mask>0]=(165,180,105)
    source=photos/'iphone.HEIC';save_heic(source,rgb)
    masks=tmp_path/'masks';masks.mkdir();Image.fromarray(mask).save(masks/'iphone.png')
    assert main([str(photos),'-o',str(seeds),'--scale-mode','manual','--um-per-pixel','.8',
                 '--mask-dir',str(masks)])==0
    m=json.loads((seeds/'manifest.json').read_text());row=m['images'][0]
    directory=seeds/'images'/row['image_id']
    assert (directory/'original.heic').read_bytes()==source.read_bytes()
    assert row['decoding']['format']=='HEIF'
    assert m['dependencies']['pillow-heif']!='unknown' and m['libheif_version']
    assert 'original-preview.png' in (seeds/'index.html').read_text()
    assert np.array_equal(np.asarray(Image.open(directory/'original-preview.png')),read_image(source))
    output=tmp_path/'tubes'
    args=argparse.Namespace(seed_report=[seeds],source_dir=None,output=output,working_width=340,
                            min_width=4.,max_width=30.,patch_side=2)
    assert run(args)==0
    row=json.loads((output/'manifest.json').read_text())['images'][0]
    directory=output/'images'/row['image_id']
    assert (directory/'original.heic').read_bytes()==source.read_bytes()
    assert row['original_preview_file']=='original-preview.png'
    html=(output/'index.html').read_text()
    assert '/original-preview.png' in html and "src='images/01-iphone/original.heic'" not in html
    assert row['proposals'][0]['methods']['ensemble']['fits'][0]['measurement_guides']['length_um']>0
    assert (output/row['png_report']).exists()


def test_corrupt_heic_does_not_hide_valid_photos(tmp_path):
    photos=tmp_path/'photos';photos.mkdir();Image.fromarray(colours()).save(photos/'valid.png')
    (photos/'bad.HEIC').write_bytes(b'not an image')
    output=tmp_path/'report'
    assert main([str(photos),'-o',str(output),'--scale-mode','manual','--um-per-pixel','.8',
                 '--segmentation-profile','generic'])==2
    m=json.loads((output/'manifest.json').read_text())
    assert [r['status']=='error' for r in m['images']]==[True,False]
    assert m['images'][0]['photo']=='bad.HEIC'
