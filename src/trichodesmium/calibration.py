"""Ruler detection with reviewable profiles; never assume 600 um is visible."""
from dataclasses import asdict, dataclass, field
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import find_peaks


@dataclass
class Calibration:
    method: str
    um_per_px: float | None = None
    quality: str = "не определена"
    reasons: list[str] = field(default_factory=list)
    axes: list[dict] = field(default_factory=list)
    tick_um: float = 3.0
    ruler_length_um: float = 600.0

    def to_dict(self):
        return asdict(self)


def positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def manual_scale(value):
    return Calibration("manual", positive(value, "um_per_px"), "средняя",
                       ["manual_scale_not_independently_verified"])


def point_scale(points, distance_um):
    p = np.asarray(points, dtype=float).reshape(2, 2)
    if not np.isfinite(p).all():
        raise ValueError("Reference coordinates must be finite")
    pixel_distance = positive(float(np.linalg.norm(p[1] - p[0])), "pixel distance")
    return Calibration("reference_points", positive(distance_um, "distance_um") / pixel_distance,
                       "средняя", ["reference_transfer_requires_same_pixel_sampling"])


def ruler_axes(rgb):
    """Find distinct long dark line candidates, including rotated rulers."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 100)
    short = min(gray.shape)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(30, short // 10),
                            minLineLength=short * .35, maxLineGap=short * .025)
    if lines is None:
        return []
    ordered = sorted(lines[:, 0], key=lambda q: -np.hypot(q[2]-q[0], q[3]-q[1]))
    distinct = []
    for q in ordered:
        p0, p1 = np.asarray(q[:2], float), np.asarray(q[2:], float)
        direction = (p1 - p0) / np.linalg.norm(p1 - p0)
        if direction[0] < 0 or (abs(direction[0]) < .01 and direction[1] < 0):
            p0, p1 = p1, p0
            direction = -direction
        duplicate = False
        for a, b in distinct:
            d = (b-a) / np.linalg.norm(b-a)
            offset=p0-a
            cross=d[0]*offset[1]-d[1]*offset[0]
            if abs(np.dot(d, direction)) > .995 and abs(cross) < 12:
                duplicate = True
                break
        if not duplicate:
            distinct.append((p0, p1))
        if len(distinct) >= 16:
            break
    return distinct


def analyse_axis(rgb, p0, p1):
    """Sample outside the continuous axis, where transverse ticks recur."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(float) / 255
    length = float(np.linalg.norm(p1-p0))
    direction = (p1-p0) / length
    normal = np.array([-direction[1], direction[0]])
    t = np.linspace(0, length, int(length)+1)
    offsets = np.concatenate((np.arange(-24, -4), np.arange(5, 25)))
    xy = p0[:, None, None] + direction[:, None, None]*t[None, None, :] + normal[:, None, None]*offsets[None, :, None]
    # Ruler ink should be dark and approximately achromatic. Pigmented cell
    # septa must not become a ruler merely because they repeat regularly.
    saturation=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)[...,1]/255
    ink=(1-gray)*((gray<.55)&(saturation<.35))
    samples = map_coordinates(ink, [xy[1], xy[0]], order=1, mode="constant", cval=0)
    profile = gaussian_filter1d(np.mean(samples, axis=0), .7)
    baseline = gaussian_filter1d(profile, 15)
    signal = profile - baseline
    prominence = max(.025, float(np.std(signal))*.7)
    peaks, _ = find_peaks(signal, prominence=prominence, distance=3)
    # Require corresponding marks on BOTH sides of the continuous axis. This
    # deliberately fails on one-sided rulers rather than confusing a rod edge
    # and its cellular texture with a microscope scale.
    profiles=[gaussian_filter1d(np.mean(side,axis=0),.7) for side in np.split(samples,2)]
    side_peaks=[]
    for side in profiles:
        side_signal=side-gaussian_filter1d(side,15)
        found,_=find_peaks(side_signal,prominence=max(.025,np.std(side_signal)*.7),distance=3)
        side_peaks.append(found)
    peaks=np.array([peak for peak in peaks if all(len(side) and np.min(abs(side-peak))<=2 for side in side_peaks)],dtype=int)
    if len(peaks) < 9:
        return None
    positions = t[peaks]
    gaps = np.diff(positions)
    spacing = float(np.median(gaps))
    if spacing < 4:
        return None
    # Missing ticks are allowed at integer multiples; fundamental spacing must
    # still occur in most intervals. No arbitrary division by two/three.
    multiples = np.clip(np.round(gaps/spacing), 1, 5)
    residual = np.abs(gaps / spacing - multiples)
    consistent = float(np.mean(residual < .15))
    fundamental = float(np.mean(np.abs(gaps/spacing - 1) < .15))
    if consistent < .8 or fundamental < .6:
        return None
    refined = float(np.median(gaps[residual < .15] / multiples[residual < .15]))
    return {"p0_xy": p0.tolist(), "p1_xy": p1.tolist(), "spacing_px": refined,
            "tick_positions_px": positions.tolist(), "regular_fraction": consistent,
            "fundamental_fraction": fundamental, "profile": profile.tolist()}


def auto_scale(rgb, tick_um=3., ruler_length_um=600.):
    positive(tick_um, "tick_um")
    positive(ruler_length_um, "ruler_length_um")
    valid = []
    for p0, p1 in ruler_axes(rgb):
        axis = analyse_axis(rgb, p0, p1)
        if axis:
            valid.append(axis)
    # Candidate axis detection alone is not evidence of a ruler.
    result = Calibration("auto", tick_um=tick_um, ruler_length_um=ruler_length_um)
    if not valid:
        result.reasons = ["no_regular_ruler_ticks"]
        return result
    valid.sort(key=lambda a: (-a["regular_fraction"], -len(a["tick_positions_px"])))
    primary = valid[0]
    # Repeated spacing along nonparallel axes is the additional validation.
    d0 = np.subtract(primary["p1_xy"], primary["p0_xy"])
    others = [a for a in valid[1:] if abs(np.dot(d0/np.linalg.norm(d0),
              np.subtract(a["p1_xy"], a["p0_xy"])/np.linalg.norm(np.subtract(a["p1_xy"], a["p0_xy"])))) < .9]
    result.axes = [primary] + others[:1]
    if others:
        relative_error = abs(others[0]["spacing_px"]/primary["spacing_px"] - 1)
        if relative_error > .08:
            result.reasons = ["ruler_axes_disagree", "possible_anisotropic_resize_or_wrong_axis"]
            return result
    result.um_per_px = tick_um / float(np.mean([a["spacing_px"] for a in result.axes]))
    result.quality = "средняя" if others else "низкая"
    result.reasons = ["experimental_ruler_detection_requires_review", "full_600_um_endpoints_not_verified"]
    if not others:
        result.reasons.append("only_one_ruler_axis")
    return result


def ruler_mask(shape, calibration):
    """Mask detected axes/ticks for flags and exclusion of width/septum samples.

    Do NOT erase this mask from object segmentation: overlaps are real missing data.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for axis in calibration.axes:
        p0, p1 = np.asarray(axis["p0_xy"]), np.asarray(axis["p1_xy"])
        direction = (p1-p0)/np.linalg.norm(p1-p0)
        normal = np.array([-direction[1], direction[0]])
        cv2.line(mask, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), 1, 10)
        for t in axis["tick_positions_px"]:
            center = p0 + direction*t
            a, b = center - normal*25, center + normal*25
            cv2.line(mask, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 1, 5)
    return mask.astype(bool)
