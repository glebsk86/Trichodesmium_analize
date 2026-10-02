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


def field_mask(rgb):
    """Exclude the black camera surround, without filling holes in object masks."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bright = gray > 50
    labels = label(bright)
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    if not areas.any():
        return np.zeros(gray.shape, bool)
    # The microscope illumination is the largest connected bright region.
    field = binary_fill_holes(labels == np.argmax(areas))
    return cv2.erode(field.astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(bool)


def candidates(rgb, min_area=80, contrast=.08, saturation=.10):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    # Slow-varying illumination correction; exact operating thresholds need data.
    background = cv2.GaussianBlur(gray, (0, 0), 20)
    relative_dark = (background - gray)/np.maximum(background, .05)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    local_saturation = hsv[..., 1].astype(np.float32)/255
    saturation_background = cv2.GaussianBlur(local_saturation, (0, 0), 20)
    # A yellow/beige field is not itself a pigmented organism.
    colorful = (local_saturation-saturation_background > saturation) & (gray < .85)
    mask = ((relative_dark > contrast) | colorful) & field_mask(rgb)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    # Global filling used to turn chamber outlines into huge solid objects.
    labels=label(mask,connectivity=2)
    areas=np.bincount(labels.ravel())
    small=areas<min_area
    small[0]=True
    labels[small[labels]]=0
    return label(labels>0, connectivity=2).astype(np.int32)


def read_labels(path, shape, mode="auto"):
    """Binary masks or integer instance masks in EXIF-oriented image coordinates."""
    with Image.open(path) as im:
        data = np.asarray(im)
    # Browser canvas exports RGB/RGBA PNG, even when the IDs are grayscale.
    if data.ndim == 3 and data.shape[2] in (3, 4):
        if not np.all(data[..., :3] == data[..., :1]):
            raise ValueError("Mask RGB channels must contain identical grayscale IDs")
        if data.shape[2] == 4 and not np.all(data[..., 3] == 255):
            raise ValueError("Mask alpha must be fully opaque")
        data = data[..., 0]
    if data.ndim != 2 or data.shape != tuple(shape[:2]):
        raise ValueError("Mask must be grayscale and match EXIF-oriented photograph size")
    if not np.issubdtype(data.dtype, np.integer):
        raise ValueError("Mask must contain integer object IDs")
    values = np.unique(data)
    if mode not in ("auto", "binary", "instances"):
        raise ValueError("Mask mode must be auto, binary or instances")
    if mode == "binary" or (mode == "auto" and len(values) <= 2):
        return label(data > 0, connectivity=2).astype(np.int32)
    return data.astype(np.int32)


def list_images(directory, recursive=False):
    paths = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in paths if p.is_file() and p.suffix.lower() in EXTENSIONS)
