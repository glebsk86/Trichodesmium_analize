"""Reviewable A/B experiment on frozen segmentation proposals, not morphometry."""
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

from .cli import provenance
from .calibration import Calibration, ruler_mask
from .imaging import field_mask, read_image
from .reporting import write_json
from .tubular import TubeSettings, compare_proposal, pair_overlap
from .tube_scale import image_scale, check_scale_consistency
from .tube_report import natural_key, build_report
from .tube_guides import measurement_guides, draw_guides, guide_label, with_header


KINDS = ("single", "ensemble")
NAMES = ("Исходное фото", "Исходная маска 0.3", "A: одна лента", "B: 7 вариантов + квадраты")
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
    # Portable fallback; the report text remains UTF-8 even without this font.
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def mask_png(path, mask):
    Image.fromarray(mask.astype(np.uint8)*255).save(path)


def metadata(fit):
    excluded = {"template", "supported", "variants"}
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
    sheet = Image.new("RGB", (width*4, height+80), "white")
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
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if args.working_width < 64:
        raise ValueError("Working width must be at least 64 pixels")
    if args.source_dir and len(args.source_dir) != len(args.seed_report):
        raise ValueError("Supply one source-dir per seed-report, in the same order")
    settings = TubeSettings(min_width_px=args.min_width, max_width_px=args.max_width,
                            patch_side_px=args.patch_side,
                            smoothness_weight=getattr(args,"smoothness_weight",3.)).validate()
    output.mkdir(parents=True)
    simple_args = argparse.Namespace(**{k: [str(p) for p in v] if isinstance(v, list) else v
                                        for k, v in vars(args).items()})
    manifest = provenance(simple_args)
    manifest.update(experiment="tube-comparison", settings=asdict(settings),
                    working_coordinates="EXIF-oriented; working-width normalization; physical scale recorded per image",
                    weight_notice="Softmax fit weights, not probabilities of Trichodesmium identity",
                    seed_reports=[], images=[], duplicates=[], errors=[])
    # Calibrate and cross-check the entire series before issuing any µm labels.
    scale_cache = {};scale_seen = set()
    for ri,report in enumerate(args.seed_report):
        seed = json.loads((report/"manifest.json").read_text(encoding="utf-8"))
        source_dir = args.source_dir[ri] if args.source_dir else Path(seed["parameters"]["input"])
        for image in seed["images"]:
            try:
                source_rgb = read_image(source_dir/image["photo"])
                digest = hashlib.sha256(source_rgb.tobytes()+str(source_rgb.shape).encode()).hexdigest()
                if digest in scale_seen:continue
                h,w = source_rgb.shape[:2]
                wh = max(1,round(h*args.working_width/w))
                scale_cache[ri,image["photo"]] = image_scale(source_rgb,image.get("calibration"),[args.working_width/w,wh/h])
                scale_seen.add(digest)
            except (OSError,ValueError,KeyError):
                pass  # Main pass preserves per-image errors.
    manifest["scale_consistency"] = check_scale_consistency(list(scale_cache.values()),getattr(args,"scale_tolerance",.10))
    seen = {}
    html_parts = ["<!doctype html><html lang='ru'><meta charset='utf-8'><title>Сравнение лент A/B</title>",
                  "<style>body{font:17px system-ui;margin:24px;background:#f4f6f6}section{background:white;padding:20px;margin:24px 0}img{max-width:100%}a{color:#17646d}details{margin:10px 0}p{line-height:1.45}</style>",
                  "<h1>Сравнение двух моделей контура</h1>",
                  "<p>A: одна лента с уточнением по цвету. B: семь вариантов положения/ширины, оценка квадратов с обеих сторон края. "
                  "Красный контур — прошедшая фильтр гипотеза; серый — отклонённая; пурпурный — неоднозначное перекрытие. "
                  "Зелёным показаны только участки с двусторонней поддержкой границы. Это не экспертная разметка.</p>",
                  "<p>Маска-шаблон может проходить через разрыв. Supported-маска исключает шкалу и общие узлы. "
                  "Голубые линии — поддержанные участки оси длины, жёлтые — сечения ширины. Подписи над фото используют рабочие пиксели и мкм при доступной калибровке. "
                  "L участка — сумма поддержанных отрезков, W маски — среднее нарисованных сечений подобранной маски, не проверенная внешняя ширина клеток. "
                  "Серая штриховая ось обозначает только гипотезу в пробелах. Калибровка в мкм экспериментальная и требует проверки. "
                  "Число осей не равно числу трихомов; пропуски исходного детектора этим опытом не исправляются.</p>"]
    for report_index, report in enumerate(args.seed_report):
        report = report.expanduser().resolve()
        seed_manifest = json.loads((report/"manifest.json").read_text(encoding="utf-8"))
        source_dir = args.source_dir[report_index] if args.source_dir else Path(seed_manifest["parameters"]["input"])
        source_dir = source_dir.expanduser().resolve()
        manifest["seed_reports"].append({"path": str(report), "manifest_sha256": sha256(report/"manifest.json"),
                                         "source_revision": seed_manifest.get("git_commit"),
                                         "source_version": seed_manifest.get("program_version")})
        for image in sorted(seed_manifest["images"],key=lambda im:natural_key(im["photo"])):
            try:
                source = source_dir/image["photo"]
                rgb_original = read_image(source)
                decoded_hash = hashlib.sha256(rgb_original.tobytes()+str(rgb_original.shape).encode()).hexdigest()
                if decoded_hash in seen:
                    manifest["duplicates"].append({"photo": image["photo"], "duplicate_of": seen[decoded_hash],
                                                   "file_sha256": sha256(source), "decoded_rgb_sha256": decoded_hash})
                    continue
                file_hash = sha256(source)
                if file_hash != image["sha256"]:
                    raise ValueError("Source photograph no longer matches seed-report SHA-256")
                folder = report/"images"/image["image_id"]
                labels = np.asarray(Image.open(folder/"labels.tif"))
                obstruction = np.asarray(Image.open(folder/"ruler_mask.png")) > 0
                if labels.shape != rgb_original.shape[:2] or obstruction.shape != labels.shape:
                    raise ValueError("Seed masks do not match EXIF-oriented photograph")
                h, w = labels.shape
                wh = max(1, round(h*args.working_width/w))
                rgb = cv2.resize(rgb_original, (args.working_width, wh), interpolation=cv2.INTER_AREA)
                labels = cv2.resize(labels.astype(np.float32), (args.working_width, wh), interpolation=cv2.INTER_NEAREST).astype(np.int32)
                obstruction = cv2.resize(obstruction.astype(np.uint8), (args.working_width, wh), interpolation=cv2.INTER_NEAREST)>0
                calibration = scale_cache[report_index,image["photo"]]
                if calibration['origin']=='recomputed_from_source_ruler_windows':
                    refined = ruler_mask(rgb_original.shape,Calibration(**calibration['source_calibration']))
                    obstruction |= cv2.resize(refined.astype(np.uint8),(args.working_width,wh),interpolation=cv2.INTER_NEAREST)>0
                field = field_mask(rgb)
                proposals = [{"source_label": int(ident),
                              "result": compare_proposal(rgb, labels == ident, field, obstruction, settings)}
                             for ident in np.unique(labels) if ident > 0]
                events = global_overlaps(proposals, settings)
                image_id = f"{len(manifest['images'])+1:02d}-{source.stem}"
                dest = output/"images"/image_id
                dest.mkdir(parents=True)
                shutil.copy2(source, dest/("original"+source.suffix.lower()))
                Image.fromarray(rgb).save(dest/"working-original.png")
                Image.fromarray(labels).save(dest/"seed-labels.tif",compression="tiff_deflate")
                mask_png(dest/"obstruction.png", obstruction)
                layers = [rgb.copy() for _ in range(4)]
                contour(layers[1], labels > 0, (225, 35, 15))
                supported_layers = [rgb.copy(), rgb.copy()]
                guide_layers = [rgb.copy(),rgb.copy()]
                guide_labels = [[],[]]
                row = {"photo": image["photo"], "image_id": image_id, "source_path": str(source),
                       "file_sha256": file_hash, "decoded_rgb_sha256": decoded_hash,
                       "seed_labels_sha256": sha256(folder/"labels.tif"),
                       "seed_obstruction_sha256": sha256(folder/"ruler_mask.png"),
                       "source_report": report_index, "source_image_id": image["image_id"],
                       "original_size_xy": [w, h], "working_size_xy": [args.working_width, wh],
                       "original_to_working_scale_xy": [args.working_width/w, wh/h],
                       "cross_proposal_overlaps": events, "proposals": [],
                       "original_file":"original"+source.suffix.lower(), "calibration":calibration}
                html_parts.extend([f"<section><h2>{html.escape(image['photo'])}</h2>",
                                   f"<a href='images/{image_id}/comparison.jpg'><img src='images/{image_id}/comparison.jpg'></a>"])
                for proposal in proposals:
                    ident = proposal["source_label"]
                    result = proposal["result"]
                    propdir = dest/f"proposal-{ident:02d}"
                    propdir.mkdir()
                    source_mask = labels == ident
                    y, x = np.nonzero(source_mask)
                    bbox = [max(0,int(x.min())-25),max(0,int(y.min())-25),
                            min(labels.shape[1],int(x.max())+26),min(labels.shape[0],int(y.max())+26)]
                    x0,y0,x1,y1 = bbox
                    Image.fromarray(rgb[y0:y1,x0:x1]).save(propdir/"original-crop.png")
                    mask_png(propdir/"seed-mask.png",source_mask)
                    item = {"source_label": ident, "crop_bbox_working_xyxy": bbox,
                            "axis_cases": result["axis_cases"],
                            "template_gap_links_xy": result["template_gap_links_xy"], "methods": {}}
                    for k, kind in enumerate(KINDS):
                        method = result["methods"][kind]
                        md = propdir/kind
                        md.mkdir()
                        mask_png(md/"ambiguous.png",method["ambiguous_mask"])
                        for j,fit in enumerate(method["fits"],1):
                            guides=export_fit(md/f"spine-{j:02d}",fit,rgb,f"{ident}.{j}",calibration)
                            fit["measurement_guides"]=guides
                            draw_guides(guide_layers[k],fit,guides)
                            contour(guide_layers[k],fit["template"],(225,35,15) if not fit["reasons"] else (150,150,150))
                            guide_labels[k].append(guide_label(f"{ident}.{j}",guides))
                            color = (225,35,15) if not fit["reasons"] else (150,150,150)
                            contour(layers[k+2],fit["template"],color)
                            contour(supported_layers[k],fit["supported"],(20,230,95),2)
                            xy = np.rint(fit["axis_xy"][[0,-1]].mean(axis=0)).astype(int)
                            cv2.putText(layers[k+2],f"{ident}.{j}",tuple(xy),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1)
                        contour(layers[k+2],method["ambiguous_mask"],(240,30,240),2)
                        item["methods"][kind] = {key: value for key, value in method.items()
                                                if key not in {"fits","crossing_mask","ambiguous_mask"}}
                        item["methods"][kind]["fits"] = [{key:fit[key] for key in
                            ("status","reasons","supported_fraction","median_proposed_width_px",
                             "template_axis_length_px","raw_width_relative_mad","curvature_width_95",
                             "objective","selected_variant","base_objective","smoothness_bonus",
                             "boundary_regularity","boundary_bending_cost","boundary_waviness_cost",
                             "measurement_guides")} for fit in method["fits"]]
                        status = STATUS.get(method["status"],method["status"])
                        html_parts.append(f"<details><summary>Объект {ident}, {'A' if k==0 else 'B'}: {status}; "
                                          f"осей {len(method['fits'])}</summary><p>{html.escape(str(method['reasons']))}</p>")
                        for j,fit in enumerate(method["fits"],1):
                            rel = f"images/{image_id}/proposal-{ident:02d}/{kind}/spine-{j:02d}"
                            html_parts.append(f"<p>Ось {ident}.{j}: {STATUS[fit['status']]}; "
                                              f"{html.escape(', '.join(fit['reasons']))} · "
                                              f"<a href='{rel}/template.png'>Шаблон</a> · "
                                              f"<a href='{rel}/supported.png'>Поддержанные участки</a> · "
                                              f"<a href='{rel}/details.json'>Ось, оценки, веса вариантов</a></p>"
                                              f"<a href='{rel}/measurement_overlay.jpg'><img src='{rel}/measurement_overlay.jpg' "
                                              f"style='max-height:650px' alt='Ось длины и линии ширины {ident}.{j}'></a>")
                        html_parts.append("</details>")
                    write_json(propdir/"details.json",item)
                    row["proposals"].append(item)
                comparison_sheet(layers,image["photo"]).save(dest/"comparison.jpg",quality=95,subsampling=0)
                for k,kind in enumerate(KINDS):
                    Image.fromarray(layers[k+2]).save(dest/(kind+"-templates.jpg"),quality=95,subsampling=0)
                    Image.fromarray(supported_layers[k]).save(dest/(kind+"-supported.jpg"),quality=95,subsampling=0)
                    annotated,header=with_header(guide_layers[k],guide_labels[k])
                    annotated.save(dest/(kind+"-measurements.jpg"),quality=95,subsampling=0)
                    row.setdefault("measurement_overlay_header_px",{})[kind]=header
                html_parts.append(f"<p>Участки с поддержкой двух краёв: "
                                  f"<a href='images/{image_id}/single-supported.jpg'>A</a> · "
                                  f"<a href='images/{image_id}/ensemble-supported.jpg'>B</a></p>")
                html_parts.append(f"<p>Оси длины и линии ширины с подписями: "
                                  f"<a href='images/{image_id}/single-measurements.jpg'>A</a> · "
                                  f"<a href='images/{image_id}/ensemble-measurements.jpg'>B</a></p>")
                html_parts.append("</section>")
                write_json(dest/"details.json",row)
                manifest["images"].append(row)
                seen[decoded_hash] = image["photo"]
                print(f"{image['photo']}: {len(proposals)} common proposals",flush=True)
            except (OSError, ValueError, KeyError) as error:
                manifest["errors"].append({"photo": image.get("photo"), "error": str(error)})
                print(f"ERROR {image.get('photo')}: {error}",flush=True)
    summary = {"unique_images": len(manifest["images"]), "excluded_duplicate_images": len(manifest["duplicates"]),
               "common_proposals": sum(len(p["proposals"]) for p in manifest["images"]), "methods": {}}
    for kind in KINDS:
        methods = [obj["methods"][kind] for im in manifest["images"] for obj in im["proposals"]]
        fits = [fit for method in methods for fit in method["fits"]]
        summary["methods"][kind] = {"proposal_statuses": dict(Counter(m["status"] for m in methods)),
                                   "axis_hypotheses": len(fits),
                                   "axes_passing_filter": sum(not f["reasons"] for f in fits),
                                   "axes_rejected": sum(bool(f["reasons"]) for f in fits),
                                   "rejection_reasons": dict(Counter(r for f in fits for r in f["reasons"])),
                                   "median_source_coverage": float(np.median([m["source_coverage"] for m in methods])) if methods else None}
    summary["calibrated_images"] = sum(im["calibration"]["working_um_per_px_xy"] is not None for im in manifest["images"])
    summary["notice"] = "Internal filtering counts, not biological recall/precision. Source coverage measures old proposal agreement, not truth."
    manifest["summary"] = summary
    build_report(output,manifest)
    write_json(output/"manifest.json",manifest)
    write_json(output/"summary.json",summary)
    (output/"comparison.html").write_text("\n".join(html_parts)+"</html>",encoding="utf-8")
    return 1 if manifest["errors"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-report",type=Path,action="append",required=True,
                        help="Frozen regular CLI report; repeat for more reports")
    parser.add_argument("--source-dir",type=Path,action="append",help="Relocated photo directory, one per seed-report")
    parser.add_argument("--output","-o",type=Path,required=True)
    parser.add_argument("--working-width",type=int,default=960)
    parser.add_argument("--min-width",type=float,default=4.)
    parser.add_argument("--max-width",type=float,default=30.)
    parser.add_argument("--patch-side",type=int,default=2)
    parser.add_argument("--scale-tolerance",type=float,default=.10,help="Warn above this relative scale deviation within identical original resolutions")
    parser.add_argument("--smoothness-weight",type=float,default=3.,
                        help="B-only bonus for regular side boundaries; 0 reproduces 0.4 ranking")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError,ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
