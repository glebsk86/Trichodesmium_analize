import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from trichodesmium.local_app import main


def test_one_command_emits_b_values_tables_and_refuses_overwrite(tmp_path):
    photos=tmp_path/'photos';photos.mkdir();masks=tmp_path/'masks';masks.mkdir()
    mask=np.zeros((180,340),np.uint8);cv2.line(mask,(30,90),(300,90),1,10)
    rgb=np.full((*mask.shape,3),(195,195,180),np.uint8);rgb[mask>0]=(165,180,105)
    Image.fromarray(rgb).save(photos/'rod.png');Image.fromarray(mask).save(masks/'rod.png')
    output=tmp_path/'result'
    args=[str(photos),'-o',str(output),'--scale-mode','manual','--um-per-pixel','.8',
          '--mask-dir',str(masks),'--working-width','340']
    assert main(args)==0
    info=json.loads((output/'run-info.json').read_text())
    assert info['rows']==1 and info['summary']['calibrated_images']==1
    manifest=json.loads((output/'report/manifest.json').read_text())
    fit=manifest['images'][0]['proposals'][0]['methods']['ensemble']['fits'][0]
    with (output/'B-measurements.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.reader(stream,delimiter=';'))
    assert len(rows)==2
    assert float(rows[1][3])==fit['measurement_guides']['length_um']
    assert float(rows[1][4])==fit['measurement_guides']['width_um']
    assert rows[1][-3:]==['не определён','','']
    assert (output/'B-measurements.xlsx').exists()
    assert 'report/index.html' in (output/'index.html').read_text()
    assert main(args)==2


def test_missing_scale_still_emits_pixel_report_and_corrupt_file_is_explicit(tmp_path):
    photos=tmp_path/'photos';photos.mkdir();Image.new('RGB',(160,100),'white').save(photos/'empty.png')
    (photos/'bad.heic').write_bytes(b'broken')
    output=tmp_path/'result'
    assert main([str(photos),'-o',str(output),'--working-width','160'])==1
    report=json.loads((output/'report/manifest.json').read_text())
    assert len(report['errors'])==1 and report['errors'][0]['photo']=='bad.heic'
    assert len(report['images'])==1
    assert report['images'][0]['calibration']['working_um_per_px_xy'] is None
    assert 'Ничего не найдено' in (output/'report/index.html').read_text()
    assert (output/'B-measurements.csv').exists()


def test_output_inside_input_is_rejected_before_writing(tmp_path):
    photos=tmp_path/'photos';photos.mkdir()
    assert main([str(photos),'-o',str(photos/'results')])==2
    assert not (photos/'results').exists()


def test_documented_module_entry_prints_help():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-m','trichodesmium.local_app','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--working-width' in result.stdout
