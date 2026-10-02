"""Capture-specific pixel proposals. Votes are not calibrated probabilities."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from .imaging import field_mask

PROFILE = "microscope-20261002"
ASSETS = Path(__file__).with_name("profiles")


@lru_cache(maxsize=1)
def load_profile():
    path = ASSETS / "microscope_20261002.xml.gz"
    metadata = json.loads(path.with_name("microscope_20261002.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata["model_sha256"]:
        raise ValueError("Segmentation profile checksum mismatch")
    model = cv2.ml.RTrees_load(str(path))
    if not model.isTrained():
        raise ValueError("Segmentation profile is not trained")
    return model, metadata


def pixel_features(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # Field normalization prevents black corners from distorting local colour.
    from scipy.ndimage import binary_fill_holes
    field = binary_fill_holes(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > 50).astype(np.float32)
    channels = [lab[..., 0]/255, (lab[..., 1]-128)/30, (lab[..., 2]-128)/50]
    for sigma in (1, 3, 10, 25):
        weights = cv2.GaussianBlur(field, (0, 0), sigma)
        mean = cv2.GaussianBlur(lab*field[..., None], (0, 0), sigma)
        mean /= np.maximum(weights[..., None], .001)
        channels.extend([(lab[..., i]-mean[..., i])/30 for i in range(3)])
    for sigma in (3, 7):
        luminance = lab[..., 0]/255
        mean = cv2.GaussianBlur(luminance, (0, 0), sigma)
        variance = cv2.GaussianBlur(luminance*luminance, (0, 0), sigma)-mean*mean
        channels.append(np.sqrt(np.maximum(variance, 0)))
    return np.stack(channels, axis=-1)


def microscope_candidates(rgb, min_area=80, min_elongation=3.):
    model, metadata = load_profile()
    features = pixel_features(rgb)
    flat = features.reshape(-1, features.shape[-1])
    votes = []
    # Bound the vote matrix memory without changing predictions at tile edges.
    for start in range(0, len(flat), 100_000):
        batch = model.getVotes(flat[start:start+100_000], 0)
        positive_column = int(np.flatnonzero(batch[0] == 1)[0])
        votes.append(batch[1:, positive_column]/batch[1:].sum(axis=1))
    score = np.concatenate(votes).reshape(rgb.shape[:2])
    mask = ((score > .55) & field_mask(rgb)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    proposals = label(mask, connectivity=2)
    output = np.zeros(mask.shape, np.int32)
    accepted = 0
    for region in regionprops(proposals):
        if region.area < max(250, min_area):
            continue
        elongation = region.axis_major_length/max(region.axis_minor_length, 1.)
        if elongation < min_elongation:
            continue
        skeleton = skeletonize(region.image)
        width = 2*float(np.median(distance_transform_edt(region.image)[skeleton]))
        extent = int(skeleton.sum())
        # Debris rejection, not an imposed limit on the number of objects.
        if extent < 80 or width > 30 or extent/max(width, 1.) < 6:
            continue
        accepted += 1
        y0, x0, y1, x1 = region.bbox
        view = output[y0:y1, x0:x1]
        view[region.image] = accepted
    info = {"profile": PROFILE, "model_sha256": metadata["model_sha256"],
            "training_notice": metadata["training_notice"],
            "vote_threshold": .55, "opening_px": 3, "closing_px": 7,
            "minimum_medial_extent_px": 80, "maximum_median_core_width_px": 30,
            "minimum_area_px": max(250, min_area),
            "proposal_count_before_filtering": int(proposals.max()),
            "source_dimensions_match_training": [rgb.shape[1], rgb.shape[0]] == metadata["source_dimensions_px"],
            "notice": "Disconnected detections can be pieces of one filament. Core width is not a reviewed outer boundary."}
    return output, info
