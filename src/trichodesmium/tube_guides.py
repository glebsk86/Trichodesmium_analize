"""Auditable lines for supported-axis length and fitted-mask width in pixels."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fonts import report_font
from .tubular import normals, sample


def measurement_guides(fit, width_samples=12, calibration=None):
    """Never bridge unsupported pixels or issue measurements for rejected fits.

    W is the fitted mask's width, not an independently measured cell envelope.
    L is the sum of supported axis intervals, not a reconstructed whole chain.
    Coordinates and units use the working image, with its transform in manifest.
    """
    xy = fit["axis_xy"]
    result = {"units":"working_px", "coordinate_space":"working EXIF-oriented image",
              "visible_axis_length_px":None, "mean_fitted_width_px":None,
              "full_chain_length_px":None, "length_um":None, "width_um":None,
              "length_segments_xy":[], "length_segments_px":[],
              "width_lines_xy":[], "width_samples_px":[],
              "length_segments_um":[], "width_samples_um":[], "calibration":calibration,
              "notice":"L sums supported axis intervals; W is fitted-mask width. Not verified biological dimensions."}
    if fit["reasons"] or len(xy)<2:
        return result
    supported = fit["supported"]
    steps = np.diff(xy,axis=0)
    # Check several points on EVERY interval so a one-pixel gap is not bridged.
    probes = xy[:-1,None,:]+np.array([0.,.25,.5,.75,1.])[None,:,None]*steps[:,None,:]
    valid = np.all(sample(supported,probes,0)>.5,axis=1)
    lengths = np.linalg.norm(steps,axis=1)
    result["length_segments_xy"] = np.stack([xy[:-1][valid],xy[1:][valid]],axis=1).tolist()
    result["length_segments_px"] = lengths[valid].tolist()
    if valid.any():
        result["visible_axis_length_px"] = float(lengths[valid].sum())
    n = normals(xy)
    left,right = fit["left_radius_px"],fit["right_radius_px"]
    margin = float(np.median(left+right))/2
    arc = np.r_[0.,np.cumsum(lengths)]
    starts,ends = xy-left[:,None]*n,xy+right[:,None]*n
    # Samples in the inner 90% must belong to this supported instance. Avoid
    # round caps and crossing/ruler pixels; only these drawn lines enter W.
    rays = starts[:,None,:]+np.linspace(.05,.95,19)[None,:,None]*(ends-starts)[:,None,:]
    eligible = ((arc>=margin)&(arc<=arc[-1]-margin)&
                np.all(sample(supported,rays,0)>.5,axis=1)&
                (sample(supported,xy,0)>.5))
    indices = np.flatnonzero(eligible)
    if len(indices):
        selected = np.unique(indices[np.rint(np.linspace(0,len(indices)-1,min(width_samples,len(indices)))).astype(int)])
        lines = np.stack([starts[selected],ends[selected]],axis=1)
        widths = np.linalg.norm(lines[:,1]-lines[:,0],axis=1)
        result["width_lines_xy"] = lines.tolist()
        result["width_samples_px"] = widths.tolist()
        result["mean_fitted_width_px"] = float(widths.mean())
    factors = (calibration or {}).get("working_um_per_px_xy")
    if factors is not None:
        factors = np.asarray(factors,dtype=float)
        if factors.shape!=(2,) or not np.isfinite(factors).all() or np.any(factors<=0):
            raise ValueError("Physical scale must contain two positive finite factors")
        for field,values,output in (("length_segments_xy","length_segments_um","length_um"),
                                    ("width_lines_xy","width_samples_um","width_um")):
            lines = np.asarray(result[field])
            if len(lines):
                distances = np.linalg.norm((lines[:,1]-lines[:,0])*factors,axis=1)
                result[values] = distances.tolist()
                result[output] = float(distances.sum() if field=="length_segments_xy" else distances.mean())
        result["units"] = "working_px_and_um"
    return result


def draw_guides(canvas,fit,guides):
    xy = np.rint(fit["axis_xy"]).astype(np.int32)
    # Dashed full axis denotes the template only; no length is counted on gaps.
    for i in range(0,len(xy)-1,4):
        cv2.line(canvas,tuple(xy[i]),tuple(xy[min(i+1,len(xy)-1)]),(150,150,150),1)
    for segment in guides["length_segments_xy"]:
        a,b = np.rint(segment).astype(np.int32)
        cv2.line(canvas,tuple(a),tuple(b),(0,215,255),2)
    for line in guides["width_lines_xy"]:
        a,b = np.rint(line).astype(np.int32)
        cv2.line(canvas,tuple(a),tuple(b),(255,205,0),2)


def guide_label(ident,guides):
    length,width = guides["visible_axis_length_px"],guides["mean_fitted_width_px"]
    ls = "—" if length is None else f"{length:.1f}"
    ws = "—" if width is None else f"{width:.1f}"
    lu,wu = guides["length_um"],guides["width_um"]
    ls = f"{lu:.1f} мкм ({ls} px)" if lu is not None else f"{ls} px"
    ws = f"{wu:.1f} мкм ({ws} px)" if wu is not None else f"{ws} px"
    calibrated = (guides.get("calibration") or {}).get("working_um_per_px_xy") is not None
    notice = "" if lu is not None or wu is not None else "; измерение не выдано" if calibrated else "; масштаб не определён"
    return f"{ident}: L участка = {ls}; W маски = {ws}; сечений {len(guides['width_samples_px'])}{notice}"


def with_header(rgb,labels,legend="Голубая ось: L участка; жёлтые сечения: W маски"):
    font = report_font(16)
    lines = []
    for text in [legend,*labels]:
        current = ""
        for word in text.split():
            trial = current+" "+word if current else word
            if current and font.getlength(trial)>rgb.shape[1]-20:
                lines.append(current);current=word
            else:current=trial
        lines.append(current)
    height = 12+24*len(lines)
    image = Image.new("RGB",(rgb.shape[1],rgb.shape[0]+height),"white")
    image.paste(Image.fromarray(rgb),(0,height))
    draw = ImageDraw.Draw(image)
    for i,line in enumerate(lines):
        draw.text((10,5+24*i),line,font=font,fill="black")
    return image,height
