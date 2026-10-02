"""Batch CLI; failed calibration keeps pixel measurements and explicit status."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from skimage.measure import regionprops

from . import __version__
from .calibration import Calibration, auto_scale, manual_scale, point_scale, positive, ruler_mask
from .geometry import measure
from .imaging import candidates, list_images, read_image, read_labels
from .segmentation import PROFILE, microscope_candidates
from .reporting import DETAIL_FIELDS, SUMMARY_FIELDS, per_photo, save_visuals, summary, workbook, write_csv, write_json, review_index


def parser():
    p=argparse.ArgumentParser(description="Experimental Trichodesmium morphometry; inspect all candidates.")
    p.add_argument("input",type=Path,help="Directory of photographs")
    p.add_argument("--output","-o",type=Path,required=True,help="New output directory; existing directories are never overwritten")
    p.add_argument("--recursive",action="store_true")
    p.add_argument("--scale-mode",choices=["auto","manual","reference"],default="auto")
    p.add_argument("--um-per-pixel",type=float)
    p.add_argument("--tick-um",type=float,default=3.)
    p.add_argument("--ruler-length-um",type=float,default=600.,help="Recorded full ruler length; not assumed visible")
    p.add_argument("--reference",type=Path)
    p.add_argument("--reference-points",type=float,nargs=4,metavar=("X1","Y1","X2","Y2"))
    p.add_argument("--reference-distance-um",type=float)
    p.add_argument("--mask-dir",type=Path,help="Optional grayscale object masks, relative path matching input but suffix .png")
    p.add_argument("--mask-format",choices=["auto","binary","instances"],default="auto",
                   help="Use instances for masks downloaded from the review editor")
    p.add_argument("--segmentation-profile",choices=[PROFILE,"generic"],default=PROFILE,
                   help="Capture-specific experimental core detector or generic local-contrast baseline")
    p.add_argument("--min-area",type=int,default=80)
    p.add_argument("--min-elongation",type=float,default=3.)
    p.add_argument("--contrast",type=float,default=.08)
    p.add_argument("--saturation",type=float,default=.10)
    p.add_argument("--no-crops",action="store_true")
    p.add_argument("--version",action="version",version=__version__)
    return p


def provenance(args):
    root=Path(__file__).resolve().parents[2]
    revision=None
    dirty=None
    try:
        revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],stderr=subprocess.DEVNULL,text=True).strip()
        dirty=bool(subprocess.check_output(["git","-C",str(root),"status","--porcelain"],stderr=subprocess.DEVNULL,text=True).strip())
    except (OSError,subprocess.CalledProcessError): pass
    dependencies={}
    for name in ["numpy","scipy","opencv-python-headless","scikit-image","Pillow","openpyxl"]:
        try: dependencies[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: dependencies[name]="unknown"
    return {"program_version":__version__,"git_commit":revision,"git_dirty":dirty,
            "started_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
            "dependencies":dependencies,"parameters":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            "notice":"All objects and septa are unverified candidates. Species identification is not implemented."}


def validate(args):
    args.input=args.input.expanduser().resolve()
    args.output=args.output.expanduser().resolve()
    if not args.input.is_dir(): raise ValueError("Input must be a directory")
    if args.output.exists(): raise ValueError("Output already exists; choose a new directory")
    if args.output.is_relative_to(args.input): raise ValueError("Output must be outside input to prevent analyzing generated files")
    if args.min_area<5: raise ValueError("min-area must be at least 5")
    positive(args.min_elongation,"min-elongation")
    positive(args.tick_um,"tick-um")
    positive(args.ruler_length_um,"ruler-length-um")
    if not 0<args.contrast<1 or not 0<=args.saturation<=1:
        raise ValueError("contrast must be in (0,1); saturation in [0,1]")
    if args.scale_mode=="manual":
        if args.um_per_pixel is None: raise ValueError("manual mode requires --um-per-pixel")
        positive(args.um_per_pixel,"um-per-pixel")
    elif args.um_per_pixel is not None: raise ValueError("--um-per-pixel requires manual mode")
    reference_flags=bool(args.reference or args.reference_points or args.reference_distance_um is not None)
    if args.scale_mode!="reference" and reference_flags: raise ValueError("Reference options require reference mode")
    if args.scale_mode=="reference":
        if args.reference is None: raise ValueError("reference mode requires --reference")
        if bool(args.reference_points)!=(args.reference_distance_um is not None):
            raise ValueError("Reference points and distance must be supplied together")
    if args.mask_dir is not None:
        args.mask_dir=args.mask_dir.expanduser().resolve()
        if not args.mask_dir.is_dir(): raise ValueError("Mask directory does not exist")


def run(args):
    validate(args)
    paths=list_images(args.input,args.recursive)
    if not paths: raise ValueError("No supported photographs found (JPG, PNG, TIFF, BMP)")
    reference=None
    reference_shape=None
    if args.scale_mode=="reference":
        ref=read_image(args.reference)
        reference_shape=ref.shape
        if args.reference_points:
            p=np.array(args.reference_points).reshape(2,2)
            if np.any(p<0) or np.any(p[:,0]>=ref.shape[1]) or np.any(p[:,1]>=ref.shape[0]):
                raise ValueError("Reference points lie outside the EXIF-oriented image")
            reference=point_scale(args.reference_points,args.reference_distance_um)
        else:
            reference=auto_scale(ref,args.tick_um,args.ruler_length_um)
            reference.method="reference_auto"
        if reference.um_per_px is None: raise ValueError("Reference calibration failed; supply points and a known distance")
    args.output.mkdir(parents=True)
    manifest=provenance(args)
    rows,photos=[],[]
    errors=0
    for number,path in enumerate(paths,1):
        relative=path.relative_to(args.input)
        image_id=re.sub(r"[^\w.-]","_",path.stem)[:60]+"_"+hashlib.sha256(relative.as_posix().encode()).hexdigest()[:8]
        directory=args.output/"images"/image_id
        directory.mkdir(parents=True)
        info={"photo":relative.as_posix(),"image_id":image_id,"status":"processing","candidate_count":0,"error":None}
        print(f"[{number}/{len(paths)}] {relative}",flush=True)
        try:
            rgb=read_image(path)
            info["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            info["oriented_size_px"]=[rgb.shape[1],rgb.shape[0]]
            auto=auto_scale(rgb,args.tick_um,args.ruler_length_um)
            info["ruler_detection"]=auto.to_dict()
            if args.scale_mode=="manual":
                cal=manual_scale(args.um_per_pixel)
            elif args.scale_mode=="reference":
                cal=replace(reference,reasons=list(reference.reasons),axes=[])
                if rgb.shape!=reference_shape:
                    cal.um_per_px=None
                    cal.quality="не определена"
                    cal.reasons.append("reference_image_dimensions_mismatch")
                else: cal.reasons.append("reference_transfer_requires_same_optical_and_digital_sampling")
            else: cal=auto
            cal.tick_um=args.tick_um
            cal.ruler_length_um=args.ruler_length_um
            info["calibration"]=cal.to_dict()
            if cal.um_per_px is None: errors+=1
            obstruction=ruler_mask(rgb.shape,auto)
            imported=False
            if args.mask_dir:
                mask_path=args.mask_dir/relative.with_suffix(".png")
                if not mask_path.is_file(): raise ValueError(f"Missing object mask: {relative.with_suffix('.png')}")
                labels=read_labels(mask_path,rgb.shape,args.mask_format)
                imported=True
                info["mask_sha256"]=hashlib.sha256(mask_path.read_bytes()).hexdigest()
                info["segmentation"]={"profile":"imported_mask","mask_format":args.mask_format}
            elif args.segmentation_profile==PROFILE:
                labels,info["segmentation"]=microscope_candidates(rgb,args.min_area,args.min_elongation)
            else:
                labels=candidates(rgb,args.min_area,args.contrast,args.saturation)
                info["segmentation"]={"profile":"generic","notice":"Experimental local contrast; not tuned to supplied microscope photographs"}
            image_rows=[]
            info["segmented_region_count"]=int(len(np.unique(labels))-1)
            for region in regionprops(labels):
                elongation=region.axis_major_length/max(region.axis_minor_length,1.)
                if not imported and (region.area<args.min_area or elongation<args.min_elongation): continue
                y0,x0,y1,x1=region.bbox
                # Work on padded local crops to avoid full-frame skeletonization per object.
                cx0,cy0=max(0,x0-3),max(0,y0-3)
                cx1,cy1=min(rgb.shape[1],x1+3),min(rgb.shape[0],y1+3)
                local=labels[cy0:cy1,cx0:cx1]==region.label
                metrics=measure(local,rgb[cy0:cy1,cx0:cx1],obstruction[cy0:cy1,cx0:cx1],
                                prune_spurs=not imported and args.segmentation_profile==PROFILE)
                for key in ("path_xy","septum_points_xy"):
                    metrics[key]=[[p[0]+cx0,p[1]+cy0] for p in metrics[key]]
                metrics["width_lines_xy"]=[[[p[0]+cx0,p[1]+cy0] for p in line] for line in metrics["width_lines_xy"]]
                if y0==0 or x0==0 or y1==rgb.shape[0] or x1==rgb.shape[1]:
                    if "frame_truncated_visible_fragment" not in metrics["flags"]: metrics["flags"].append("frame_truncated_visible_fragment")
                scale=cal.um_per_px
                row={**metrics,"photo":relative.as_posix(),"object_id":image_id+f"/obj_{len(image_rows)+1:04d}",
                     "source_label":int(region.label),"species":"не определён","species_quality":"не определена",
                     "status":"candidate_requires_review","scale_um_per_px":scale,
                     "calibration_method":cal.method,"calibration_quality":cal.quality,"calibration_reasons":cal.reasons,
                     "bbox_xyxy":[int(x0),int(y0),int(x1),int(y1)],"width_sample_count":len(metrics["width_samples_px"]),
                     "cell_interval_count":len(metrics["cell_intervals_px"]),
                     "segmentation_method":"imported_mask" if imported else args.segmentation_profile,
                     "ruler_overlap_fraction":float(np.mean(obstruction[cy0:cy1,cx0:cx1][local])),
                     "crop_original":None,"crop_annotated":None,"object_mask":None}
                row["flags"].append("unverified_object_identity_and_segmentation")
                if not imported and args.segmentation_profile==PROFILE:
                    row["flags"].extend(["pigment_core_boundary_requires_review","segmentation_may_split_one_filament"])
                    if not info["segmentation"]["source_dimensions_match_training"]:
                        row["flags"].append("profile_input_dimensions_differ_from_training")
                if scale is None: row["flags"].append("uncalibrated_pixel_measurements_only")
                for px,um in [("length_px","length_um"),("width_px","width_um"),("width_std_px","width_std_um"),("cell_length_px","cell_length_um")]:
                    row[um]=row[px]*scale if row[px] is not None and scale is not None else None
                image_rows.append(row)
            info["candidate_count"]=len(image_rows)
            info["excluded_by_size_or_shape_count"]=info["segmented_region_count"]-len(image_rows)
            info["status"]="review_required" if cal.um_per_px else "uncalibrated"
            save_visuals(directory,rgb,labels,image_rows,obstruction,not args.no_crops,
                         photo_name=relative.as_posix())
            per_photo(directory,image_rows,info)
            rows.extend(image_rows)
        except Exception as exc:
            errors+=1
            info.update(status="error",error=f"{type(exc).__name__}: {exc}")
            write_json(directory/"error.json",info)
            print(f"  ERROR: {info['error']}",file=sys.stderr,flush=True)
        photos.append(info)
    write_csv(args.output/"summary.csv",[summary(row) for row in rows],SUMMARY_FIELDS)
    write_csv(args.output/"details.csv",rows,DETAIL_FIELDS)
    write_csv(args.output/"photos.csv",photos,["photo","status","candidate_count","error","calibration"])
    workbook(args.output/"measurements.xlsx",rows,photos)
    review_index(args.output,rows,photos)
    manifest.update(images=photos,object_count=len(rows),issue_count=errors,finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output/"manifest.json",manifest)
    print(f"Saved {len(rows)} candidates from {len(paths)} images to {args.output}. Inspect overlays before use.")
    return 2 if errors else 0


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    try: return run(args)
    except (ValueError,OSError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        return 2
