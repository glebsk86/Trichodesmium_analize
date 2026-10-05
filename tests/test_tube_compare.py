import argparse
import hashlib
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from trichodesmium.tube_compare import run


def test_report_preserves_sources_weights_and_rejects_overwrite(tmp_path):
    photos=tmp_path/'photos';photos.mkdir()
    seeds=tmp_path/'seeds';seeds.mkdir()
    mask=np.zeros((180,340),np.uint8);cv2.line(mask,(30,90),(300,90),1,10)
    rgb=np.full((*mask.shape,3),(195,195,180),np.uint8);rgb[mask>0]=(165,180,105)
    images=[]
    for i in (1,2):
        photo=photos/f'photo-{i}.png';Image.fromarray(rgb).save(photo)
        folder=seeds/'images'/str(i);folder.mkdir(parents=True)
        Image.fromarray(mask.astype(np.int32)).save(folder/'labels.tif')
        Image.fromarray(np.zeros(mask.shape,np.uint8)).save(folder/'ruler_mask.png')
        images.append({'photo':photo.name,'image_id':str(i),'sha256':hashlib.sha256(photo.read_bytes()).hexdigest()})
    (seeds/'manifest.json').write_text(json.dumps({'parameters':{'input':str(photos)},'images':images}))
    output=tmp_path/'report'
    args=argparse.Namespace(seed_report=[seeds],source_dir=None,output=output,working_width=340,min_width=4.,max_width=30.,patch_side=2)
    assert run(args)==0
    report=json.loads((output/'manifest.json').read_text())
    assert report['summary']['unique_images']==1
    assert report['summary']['excluded_duplicate_images']==1
    assert report['settings']['patch_side_px']==2
    directory=output/'images'/report['images'][0]['image_id']
    assert (directory/'original.png').read_bytes()==(photos/'photo-1.png').read_bytes()
    details=json.loads((directory/'proposal-01/ensemble/spine-01/details.json').read_text())
    assert details['length_um'] is None and details['width_um'] is None
    assert len(details['variants'])==7
    assert sum(v['relative_fit_weight'] for v in details['variants'])==pytest.approx(1)
    assert (directory/'comparison.jpg').is_file()
    assert (directory/'ensemble-measurements.jpg').is_file()
    assert (directory/'proposal-01/ensemble/spine-01/measurement_overlay.jpg').is_file()
    guides=details['measurement_guides']
    assert guides['visible_axis_length_px'] is not None
    assert len(guides['width_lines_xy'])==len(guides['width_samples_px'])
    assert guides['mean_fitted_width_px']==pytest.approx(np.mean(guides['width_samples_px']))
    assert (output/'index.html').is_file()
    with pytest.raises(ValueError,match='already exists'):run(args)
