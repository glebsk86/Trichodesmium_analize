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
COLOUR_PROFILE = "microscope-colour-20261005"
ASSETS = Path(__file__).with_name("profiles")


def medial_width(mask, skeleton):
    # A tight bounding box can contain no background on one or more sides.
    # Explicit exterior zeros keep EDT from treating the crop as infinite.
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    return 2*float(np.median(distance[skeleton]))


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


def pixel_votes(rgb):
    """Return uncalibrated forest votes; keep prediction batching bounded."""
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
    return score, metadata


def core_labels(rgb, score, min_area, min_elongation):
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
        width = medial_width(region.image, skeleton)
        extent = int(skeleton.sum())
        # Debris rejection, not an imposed limit on the number of objects.
        if extent < 80 or width > 30 or extent/max(width, 1.) < 6:
            continue
        accepted += 1
        y0, x0, y1, x1 = region.bbox
        view = output[y0:y1, x0:x1]
        view[region.image] = accepted
    return output, int(proposals.max())


def microscope_candidates(rgb, min_area=80, min_elongation=3.):
    score, metadata = pixel_votes(rgb)
    output, proposal_count = core_labels(rgb, score, min_area, min_elongation)
    info = {"profile": PROFILE, "model_sha256": metadata["model_sha256"],
            "training_notice": metadata["training_notice"],
            "vote_threshold": .55, "opening_px": 3, "closing_px": 7,
            "minimum_medial_extent_px": 80, "maximum_median_core_width_px": 30,
            "minimum_area_px": max(250, min_area),
            "proposal_count_before_filtering": proposal_count,
            "source_dimensions_match_training": [rgb.shape[1], rgb.shape[0]] == metadata["source_dimensions_px"],
            "notice": "Disconnected detections can be pieces of one filament. Core width is not a reviewed outer boundary."}
    return output, info


@lru_cache(maxsize=1)
def colour_kernels():
    """Directional centre, dark edges and adjacent background bands."""
    y, x = np.mgrid[-24:25, -24:25].astype(np.float32)
    kernels = []
    for angle in np.arange(0, np.pi, np.pi/16):
        along = x*np.cos(angle)+y*np.sin(angle)
        across = -x*np.sin(angle)+y*np.cos(angle)
        def band(offset, sigma):
            k = np.exp(-along*along/(2*7**2)-(across-offset)**2/(2*sigma*sigma))
            return (k/k.sum()).astype(np.float32)
        for width in (6, 10, 14):
            centre = band(0, width*.22)
            left, right = band(-width/2, 1.2), band(width/2, 1.2)
            outer_left, outer_right = band(-width/2-4, 2), band(width/2+4, 2)
            kernels.append((centre-outer_left, centre-outer_right,
                            outer_left-left, outer_right-right))
    return kernels


def colour_evidence(rgb):
    """Bilateral yellow/olive contrast, with a separate paired-edge seed score.

    Empirical camera-specific Lab units, not colour-independent identification.
    Taking the lesser of the two side contrasts rejects a single colour edge.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pigment = lab[..., 2]-.7*lab[..., 1]
    luminance = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    contrast = np.zeros(luminance.shape, np.float32)
    paired = np.zeros_like(contrast)
    for lc, rc, le, re in colour_kernels():
        bilateral = np.minimum(cv2.filter2D(pigment, -1, lc),
                               cv2.filter2D(pigment, -1, rc))
        dark_edges = np.minimum(cv2.filter2D(luminance, -1, le),
                                cv2.filter2D(luminance, -1, re))
        contrast = np.maximum(contrast, bilateral)
        paired = np.maximum(paired, np.where(dark_edges > .012, bilateral, 0))
    return contrast, paired


def associate_supported_pixels(support, seeds, min_area=80, min_elongation=3.):
    """Assign nearby supported fragments an ID without filling their gaps.

    A 5px closing is used ONLY to propose membership. Measurement pixels come
    from support, with tiny isolated speckles removed. Curvature is allowed;
    min_elongation applies to intrinsic medial extent / core width here.
    """
    association = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    association |= support
    proposals = label(association, connectivity=2)
    output = np.zeros(support.shape, np.int32)
    accepted = 0
    for region in regionprops(proposals):
        y0, x0, y1, x1 = region.bbox
        observed = region.image & support[y0:y1, x0:x1]
        pieces = label(observed, connectivity=2)
        sizes = np.bincount(pieces.ravel())
        keep = sizes >= 9
        keep[0] = False
        observed = keep[pieces]
        area = int(observed.sum())
        overlap = int(np.sum(observed & seeds[y0:y1, x0:x1]))
        # Do not let a tiny seed spread through a large chamber edge network.
        if overlap < 30 or area < max(250, min_area) or area > 12*overlap:
            continue
        skeleton = skeletonize(observed)
        extent = int(skeleton.sum())
        if not extent:
            continue
        width = medial_width(observed, skeleton)
        if extent < 80 or width > 30 or extent/max(width, 1.) < max(6, min_elongation):
            continue
        accepted += 1
        output[y0:y1, x0:x1][observed] = accepted
    return output, int(proposals.max())


def colour_assisted_candidates(rgb, min_area=80, min_elongation=3.):
    score, metadata = pixel_votes(rgb)
    field = field_mask(rgb)
    core, _ = core_labels(rgb, score, min_area, min_elongation)
    contrast, paired = colour_evidence(rgb)
    supported = ((score > .55) | ((score > .12) & (contrast > 1.5))) & field
    # Legacy closing can contain inferred pixels. Use its membership to choose
    # seeds, but retain only currently supported pixels in this new profile.
    seeds = (core > 0) & supported
    added = 0
    for region in regionprops(label((paired > 6) & field, connectivity=2)):
        if region.area < max(250, min_area):
            continue
        y0, x0, y1, x1 = region.bbox
        elongation = region.axis_major_length/max(region.axis_minor_length, 1.)
        if elongation < max(4, min_elongation):
            continue
        skeleton = skeletonize(region.image)
        if skeleton.sum() < 200:
            continue
        width = medial_width(region.image, skeleton)
        if width > 16 or np.mean(score[y0:y1, x0:x1][region.image]) < .25:
            continue
        if not np.any(seeds[y0:y1, x0:x1] & region.image):
            added += 1
        seeds[y0:y1, x0:x1] |= region.image
    # No opening: it would remove the thin, weak links we are trying to retain.
    support = supported | seeds
    output, proposal_count = associate_supported_pixels(support, seeds, min_area, min_elongation)
    info = {"profile": COLOUR_PROFILE, "model_sha256": metadata["model_sha256"],
            "training_notice": metadata["training_notice"], "forest_retrained": False,
            "source_dimensions_match_training": [rgb.shape[1], rgb.shape[0]] == metadata["source_dimensions_px"],
            "parameters": {"strong_vote": .55, "weak_vote": .12, "bilateral_lab_contrast": 1.5,
                           "legacy_seed_profile": PROFILE, "legacy_seed_opening_px": 3,
                           "legacy_seed_closing_px": 7, "legacy_seed_min_pca_elongation": min_elongation,
                           "pigment_expression": "Lab_b - 0.7*Lab_a (OpenCV uint8 Lab)",
                           "seed_paired_contrast": 6, "seed_dark_edge_contrast": .012,
                           "seed_mean_vote": .25, "seed_min_extent_px": 200,
                           "seed_max_core_width_px": 16, "seed_min_pca_elongation": max(4, min_elongation),
                           "angles": 16, "kernel_px": 49, "along_sigma_px": 7,
                           "tested_widths_px": [6, 10, 14], "centre_sigma_width_fraction": .22,
                           "edge_sigma_px": 1.2, "background_sigma_px": 2,
                           "background_offset_beyond_edge_px": 4,
                           "association_closing_px": 5, "gaps_written_into_mask": False,
                           "minimum_island_area_px": 9, "minimum_seed_overlap_px": 30,
                           "maximum_growth_factor": 12, "minimum_area_px": max(250, min_area),
                           "minimum_medial_extent_px": 80, "maximum_median_core_width_px": 30,
                           "minimum_extent_width_ratio": max(6, min_elongation)},
            "new_colour_seed_count": added, "proposal_count_before_filtering": proposal_count,
            "notice": "Colour-assisted proposals tuned on this capture series, not independent validation. "
                      "Nearby fragments can share an unverified ID; their gaps are not filled. "
                      "Mask boundaries and missed short/pale filaments require review."}
    return output, info
