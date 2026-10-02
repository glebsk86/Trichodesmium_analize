"""Input decoding and replaceable baseline segmentation."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_fill_holes
from skimage.measure import label

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def read_image(path):
    with Image.open(path) as im:
        if getattr(im, "n_frames", 1) != 1:
            raise ValueError("Multipage images must be exported as separate photographs")
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def candidates(rgb, min_area=80, contrast=.08, saturation=.10):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    # Slow-varying illumination correction; exact operating thresholds need data.
    background = cv2.GaussianBlur(gray, (0, 0), 20)
    relative_dark = (background - gray)/np.maximum(background, .05)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colorful = (hsv[..., 1]/255 > saturation) & (gray < .85)
    mask = (relative_dark > contrast) | colorful
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    mask = binary_fill_holes(mask)
    labels=label(mask,connectivity=2)
    areas=np.bincount(labels.ravel())
    small=areas<min_area
    small[0]=True
    labels[small[labels]]=0
    return label(labels>0, connectivity=2).astype(np.int32)


def read_labels(path, shape):
    """Binary masks or integer instance masks in EXIF-oriented image coordinates."""
    with Image.open(path) as im:
        data = np.asarray(im)
    if data.ndim != 2 or data.shape != tuple(shape[:2]):
        raise ValueError("Mask must be grayscale and match EXIF-oriented photograph size")
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError("Mask must contain integer object IDs")
    values = np.unique(data)
    if len(values) <= 2:
        return label(data > 0, connectivity=2).astype(np.int32)
    return data.astype(np.int32)


def list_images(directory, recursive=False):
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in EXTENSIONS)
