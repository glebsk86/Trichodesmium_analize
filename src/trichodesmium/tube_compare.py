"""B export helpers and compatibility reader for earlier seed reports."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fonts import report_font
from .cli import provenance
from .calibration import Calibration, ruler_mask
from .imaging import field_mask, read_image
from .reporting import write_json
from .tubular import TubeSettings, compare_proposal, pair_overlap
from .tube_scale import image_scale, check_scale_consistency
from .tube_report import natural_key, build_report
from .tube_guides import measurement_guides, draw_guides, guide_label, with_header


KINDS = ("ensemble",)
NAMES = ("Исходное фото", "Предложения детектора", "B: 7 вариантов + квадраты")
STATUS = {
    "accepted_candidate": "кандидат прошёл фильтр",
    "rejected": "отклонён",
    "crossing_requires_review": "возможное угловое пересечение: проверить",
    "junction_requires_review": "стык концов / узел: проверить",
    "shared_axis_requires_review": "общая ось / узел: проверить",
    "partly_explained_requires_review": "объяснена только часть: проверить",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size):
    return report_font(size)


def mask_png(path, mask):
    Image.fromarray(mask.astype(np.uint8)*255).save(path)


def metadata(fit):
    excluded = {"template", "supported", "length_support", "variants"}
    return {key: value.tolist() if isinstance(value, np.ndarray) else
            [v.tolist() for v in value] if key == "scores" else value
            for key, value in fit.items() if key not in excluded}


def export_fit(directory, fit, rgb, ident, calibration=None):
    directory.mkdir()
    mask_png(directory/"template.png", fit["template"])
    mask_png(directory/"supported.png", fit["supported"])
    details = metadata(fit)
    guides = measurement_guides(fit,calibration=calibration)
    details["measurement_guides"] = guides
    canvas = rgb.copy()
    contour(canvas,fit["template"],(225,35,15) if not fit["reasons"] else (150,150,150))
    annotation = canvas.copy()
    draw_guides(canvas,fit,guides)
    yy,xx = np.nonzero(fit["template"])
    x0,y0,x1,y1 = max(0,int(xx.min())-30),max(0,int(yy.min())-30),min(rgb.shape[1],int(xx.max())+31),min(rgb.shape[0],int(yy.max())+31)
    if x1-x0<300:
        centre=(x0+x1)//2;x0=max(0,centre-150);x1=min(rgb.shape[1],x0+300)
    image,header = with_header(canvas[y0:y1,x0:x1],[guide_label(ident,guides)])
    annotation_image,_ = with_header(annotation[y0:y1,x0:x1],[f"Объект {ident}: контур гипотезы"],legend="Красный — кандидат; серый — отклонённая гипотеза")
    annotation_image.save(directory/"annotation.png")
    image.save(directory/"measurement.png")
    image.save(directory/"measurement_overlay.jpg",quality=95,subsampling=0)
    details["measurement_overlay"] = {"file":"measurement_overlay.jpg",
                                       "crop_bbox_working_xyxy":[x0,y0,x1,y1],"header_px":header}
    details["mask_coordinates"] = "working EXIF-oriented pixels; see image transform"
    details["length_um"],details["width_um"] = guides["length_um"],guides["width_um"]
    details["measurement_notice"] = "Proposed width and template axis length are model diagnostics, not biological measurements."
    details["variants"] = []
    for i, variant in enumerate(fit["variants"], 1):
        name = f"variant-{i:02d}.png"
        mask_png(directory/name, variant["template"])
        record = metadata(variant)
        record["template_file"] = name
        details["variants"].append(record)
    write_json(directory/"details.json", details)
    return guides


def contour(canvas, mask, color, thickness=1):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, color, thickness)


def comparison_sheet(layers, title, bbox=None):
    if bbox:
        x0, y0, x1, y1 = bbox
        layers = [layer[y0:y1, x0:x1] for layer in layers]
    width, height = 480, 670
    sheet = Image.new("RGB", (width*len(layers), height+80), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), title, fill="black", font=font(19))
    for i, layer in enumerate(layers):
        draw.text((i*width+10, 40), NAMES[i], fill="black", font=font(17))
        im = Image.fromarray(layer)
        im.thumbnail((width-10, height-5))
        sheet.paste(im, (i*width+(width-im.width)//2, 75))
    return sheet


def global_overlaps(proposals, settings):
    """Also examine intersections between different initial instance IDs."""
    events = {kind: [] for kind in KINDS}
    for kind in KINDS:
        active = [(p["source_label"], i+1, fit) for p in proposals
                  for i, fit in enumerate(p["result"]["methods"][kind]["fits"])
                  if not fit["reasons"]]
        for a in range(len(active)):
            for b in range(a+1, len(active)):
                pa, ia, fa = active[a]
                pb, ib, fb = active[b]
                if pa == pb:
                    continue  # Already evaluated within the proposal.
                overlap, category, angle = pair_overlap(fa, fb, settings)
                if category is None:
                    continue
                fa["supported"] &= ~overlap
                fb["supported"] &= ~overlap
                fa["length_support"] &= ~overlap
                fb["length_support"] &= ~overlap
                events[kind].append({"proposals": [pa, pb], "spines": [ia, ib],
                                     "kind": category, "angle_deg": angle,
                                     "shared_pixels": int(overlap.sum())})
                for proposal in proposals:
                    if proposal["source_label"] in (pa, pb):
                        method = proposal["result"]["methods"][kind]
                        method["ambiguous_mask"] |= overlap
                        if category == "angular_crossing":
                            method["crossing_mask"] |= overlap
                            method["status"] = "crossing_requires_review"
                        elif category == "endpoint_junction" and method["status"] != "crossing_requires_review":
                            method["status"] = "junction_requires_review"
                        elif method["status"] != "crossing_requires_review":
                            method["status"] = "shared_axis_requires_review"
    return events



def run(args):
    """Refine an existing seed report with B only; normal app uses one decode."""
    from .pipeline import write_result,finish,resized
    from time import perf_counter
    output=args.output.expanduser().resolve()
    if output.exists():raise ValueError('Output already exists; choose a new directory')
    output.mkdir(parents=True);started=perf_counter()
    settings=TubeSettings(min_width_px=args.min_width,max_width_px=args.max_width,patch_side_px=args.patch_side)
    serial=argparse.Namespace(**{k:[str(p) for p in v] if isinstance(v,list) else v for k,v in vars(args).items()})
    manifest=provenance(serial);manifest.update(algorithm='B',settings=asdict(settings),images=[],duplicates=[],errors=[])
    seen={}
    for ri,report in enumerate(args.seed_report):
        seed=json.loads((report/'manifest.json').read_text())
        source_dir=args.source_dir[ri] if args.source_dir else Path(seed['parameters']['input'])
        for image in sorted(seed['images'],key=lambda im:natural_key(im['photo'])):
            try:
                source=source_dir/image['photo'];original,decoding=read_image(source,return_metadata=True)
                digest=hashlib.sha256(original.tobytes()+str(original.shape).encode()).hexdigest()
                if digest in seen:
                    manifest['duplicates'].append({'photo':image['photo'],'duplicate_of':seen[digest]});continue
                if sha256(source)!=image['sha256']:raise ValueError('Source photograph no longer matches seed-report SHA-256')
                folder=report/'images'/image['image_id'];labels=np.asarray(Image.open(folder/'labels.tif'))
                obstruction=np.asarray(Image.open(folder/'ruler_mask.png'))>0
                if labels.shape!=original.shape[:2] or obstruction.shape!=labels.shape:raise ValueError('Seed masks do not match photograph')
                rgb=resized(original,args.working_width);h,w=labels.shape;wh,ww=rgb.shape[:2]
                labels=cv2.resize(labels.astype(np.float32),(ww,wh),interpolation=cv2.INTER_NEAREST).astype(np.int32)
                obstruction=cv2.resize(obstruction.astype(np.uint8),(ww,wh),interpolation=cv2.INTER_NEAREST)>0
                calibration=image_scale(original,image.get('calibration'),[ww/w,wh/h])
                row=write_result(output,source,image['photo'],len(manifest['images'])+1,original,rgb,decoding,labels,obstruction,calibration,settings,{'profile':'frozen_seed_report'},[])
                manifest['images'].append(row);seen[digest]=image['photo']
            except (OSError,ValueError,KeyError) as exc:manifest['errors'].append({'photo':image['photo'],'error':str(exc)})
    finish(output,manifest,started)
    return 1 if manifest['errors'] else 0
