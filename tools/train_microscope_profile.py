"""Recreate the experimental profile from the six supplied originals.

The scribbles are approximate pigment-core hints, not expert instance masks.
All six photos are training data. Do not report these photos as holdout accuracy.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from trichodesmium.imaging import read_image
from trichodesmium.segmentation import pixel_features
from scipy.ndimage import binary_fill_holes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path, help='New OpenCV XML or XML.GZ model path')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    traces = json.loads(Path(__file__).with_name('profile_scribbles.json').read_text())
    rng = np.random.default_rng(20261002)
    xx, yy = [], []
    walls_by_photo = {
        1:[([0,566],[960,574]),([0,674],[960,679])],
        2:[([0,635],[960,816])],
        4:[([342,0],[342,1280]),([465,0],[465,1280])],
        5:[([0,797],[960,797]),([0,919],[960,919])],
        6:[([0,777],[960,777]),([0,875],[960,875]),([880,0],[880,1280])]}
    for n in range(1,7):
        rgb = read_image(args.input/f'photo_{n}_2026-10-02_20-13-31.jpg')
        if rgb.shape != (1280,960,3):
            parser.error('Scribbles require the unresized 960x1280 originals')
        features = pixel_features(rgb)
        field = binary_fill_holes(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)>50)
        mask = np.zeros(field.shape,np.uint8)
        for trace in traces[str(n)]:
            cv2.polylines(mask,[np.array(trace,np.int32)],False,1,5)
        exclusion = cv2.dilate(mask,np.ones((17,17),np.uint8))
        pos = np.flatnonzero(mask)
        neg = np.flatnonzero((exclusion==0)&field)
        flat = features.reshape(-1,features.shape[-1])
        hard = neg[flat[neg,11]>.05]
        pos = rng.choice(pos,min(6000,len(pos)),replace=False)
        neg = rng.choice(neg,10000,replace=False)
        hard = rng.choice(hard,min(10000,len(hard)),replace=False)
        walls = np.zeros(field.shape,np.uint8)
        for a,b in walls_by_photo.get(n,[]):
            cv2.line(walls,a,b,1,30)
        wi = np.flatnonzero((walls>0)&(exclusion==0)&field)
        wi = rng.choice(wi,min(16000,len(wi)),replace=False)
        hard = np.r_[hard,wi]
        indices = np.r_[pos,neg,hard]
        xx.append(flat[indices])
        yy.append(np.r_[np.ones(len(pos)),np.zeros(len(neg)+len(hard))])
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(12)
    model.setMinSampleCount(10)
    model.setActiveVarCount(5)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER,24,0))
    model.train(np.vstack(xx).astype(np.float32),cv2.ml.ROW_SAMPLE,np.hstack(yy).astype(np.int32))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    model.save(str(args.output))
    print(f'Saved {args.output}. Update profile metadata/checksum explicitly before replacing a shipped profile.')


if __name__ == '__main__':
    main()
