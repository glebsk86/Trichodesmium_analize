"""Single-decode, resolution-normalized B pipeline, with independent photo jobs."""
from collections import Counter
from dataclasses import asdict
import copy
import hashlib
from pathlib import Path
import re
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from .calibration import Calibration,auto_scale,ruler_mask
from .cli import provenance
from .filaments import associate_labels
from .imaging import read_image,read_labels,list_images,field_mask
from .reporting import write_json
from .segmentation import multiscale_colour_candidates
from .physical_settings import settings_for_photo
from .tubular import TubeSettings,compare_proposal
from .tube_compare import export_fit,contour,global_overlaps,comparison_sheet
from .tube_guides import draw_guides,with_header,guide_label
from .tube_report import build_report,natural_key
from .tube_scale import windowed_fine_scale,FINE_METHOD,check_scale_consistency


def resized(rgb,width):
    h,w=rgb.shape[:2];height=max(1,round(h*width/w))
    return cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA if width<=w else cv2.INTER_CUBIC)


def transform_calibration(calibration,factors):
    cal=copy.deepcopy(calibration)
    fx,fy=factors
    def transform(record):
        for key,value in record.items():
            if key in ('p0_xy','p1_xy'):record[key]=(np.asarray(value)*[fx,fy]).tolist()
            elif key in ('spacing_px','window_start_px','normal_offset_px'):record[key]=value*fx
            elif key=='tick_positions_px':record[key]=(np.asarray(value)*fx).tolist()
            elif key=='windows':
                for window in value:transform(window)
    for axis in cal.get('axes',[]):transform(axis)
    if cal.get('um_per_px') is not None:cal['um_per_px']/=fx
    return cal


def calibrate(original,rgb,options):
    h,w=original.shape[:2];wh,ww=rgb.shape[:2];factors=[ww/w,wh/h]
    # Tick detection has its own bounded raster; normalization is independent
    # of the filename/container and is recorded for audit.
    cal_rgb=resized(original,min(w,1920))
    detected=auto_scale(cal_rgb).to_dict()
    fine=detected.get('axes') and all(a.get('spacing_method')==FINE_METHOD for a in detected['axes'])
    if not fine or detected['um_per_px'] is None:
        retry=windowed_fine_scale(cal_rgb)
        if retry['um_per_px'] is not None:detected=retry
    source_detected=transform_calibration(detected,[w/cal_rgb.shape[1],h/cal_rgb.shape[0]])
    physical=source_detected
    if options['scale_mode']=='manual':
        physical=Calibration('manual',options['um_per_pixel'],'средняя',['manual_scale_not_independently_verified']).to_dict()
    elif options['scale_mode']=='reference':
        physical=copy.deepcopy(options['reference_calibration'])
        if [w,h]!=options['reference_size_xy']:
            physical['um_per_px']=None;physical['reasons'].append('reference_image_dimensions_mismatch')
    value=physical['um_per_px']
    calibration=dict(source_calibration=physical,seed_calibration=source_detected,origin='normalized_B_pipeline',
                     source_size_xy=[w,h],source_to_working_scale_xy=factors,
                     calibration_raster_size_xy=[cal_rgb.shape[1],cal_rgb.shape[0]],
                     working_um_per_px_xy=None if value is None else (value/np.asarray(factors)).tolist(),
                     notice='Physical scale requires visual ruler review.')
    working_detected=transform_calibration(source_detected,factors)
    return calibration,ruler_mask(rgb.shape,Calibration(**working_detected))


def write_result(output,source,relative,ordinal,original,rgb,decoding,labels,obstruction,calibration,settings,segmentation,links):
    safe=re.sub(r'[^\w.-]','_',source.stem)
    image_id=f'{ordinal:02d}-{safe}-{hashlib.sha256(relative.encode()).hexdigest()[:6]}'
    dest=output/'images'/image_id;dest.mkdir(parents=True)
    original_file='original'+source.suffix.lower();shutil.copy2(source,dest/original_file)
    preview=original_file
    if decoding['format']=='HEIF':
        preview='original-preview.png';Image.fromarray(original).save(dest/preview,compress_level=1)
    Image.fromarray(rgb).save(dest/'working-original.png',compress_level=1)
    Image.fromarray(labels).save(dest/'seed-labels.tif',compression='tiff_deflate')
    Image.fromarray(obstruction.astype(np.uint8)*255).save(dest/'obstruction.png',compress_level=1)
    field=field_mask(rgb)
    proposals=[dict(source_label=int(ident),result=compare_proposal(rgb,labels==ident,field,obstruction,settings))
               for ident in np.unique(labels) if ident>0]
    events=global_overlaps(proposals,settings)
    h,w=original.shape[:2];wh,ww=rgb.shape[:2]
    row=dict(photo=relative,image_id=image_id,source_path=str(source),file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             decoded_rgb_sha256=hashlib.sha256(original.tobytes()+str(original.shape).encode()).hexdigest(),
             original_size_xy=[w,h],working_size_xy=[ww,wh],original_to_working_scale_xy=[ww/w,wh/h],
             original_file=original_file,original_preview_file=preview,decoding=decoding,calibration=calibration,
             segmentation=segmentation,fragment_associations=links,cross_proposal_overlaps=events,proposals=[])
    layers=[rgb.copy() for _ in range(3)];contour(layers[1],labels>0,(225,35,15))
    guide_canvas=rgb.copy();captions=[]
    for proposal in proposals:
        ident=proposal['source_label'];result=proposal['result'];method=result['methods']['ensemble']
        folder=dest/f'proposal-{ident:02d}';folder.mkdir();md=folder/'ensemble';md.mkdir()
        y,x=np.nonzero(labels==ident);bbox=[max(0,int(x.min())-30),max(0,int(y.min())-30),min(ww,int(x.max())+31),min(wh,int(y.max())+31)]
        item=dict(source_label=ident,crop_bbox_working_xyxy=bbox,axis_cases=result['axis_cases'],
                  template_gap_links_xy=result['template_gap_links_xy'],methods={})
        fit_records=[]
        for j,fit in enumerate(method['fits'],1):
            guides=export_fit(md/f'spine-{j:02d}',fit,rgb,f'{ident}.{j}',calibration)
            contour(layers[2],fit['template'],(225,35,15) if not fit['reasons'] else (150,150,150))
            draw_guides(guide_canvas,fit,guides);captions.append(guide_label(f'{ident}.{j}',guides))
            fields=('status','reasons','supported_fraction','median_proposed_width_px','template_axis_length_px',
                    'raw_width_relative_mad','curvature_width_95','objective','selected_variant','base_objective',
                    'smoothness_bonus','boundary_regularity','boundary_bending_cost','boundary_waviness_cost')
            fit_records.append({**{key:fit[key] for key in fields},'measurement_guides':guides})
        item['methods']['ensemble']={**{key:value for key,value in method.items() if key not in ('fits','crossing_mask','ambiguous_mask')},'fits':fit_records}
        write_json(folder/'details.json',item);row['proposals'].append(item)
    comparison_sheet(layers,relative).save(dest/'diagnostics.jpg',quality=90)
    annotated,header=with_header(guide_canvas,captions);annotated.save(dest/'B-measurements.png',compress_level=1)
    row['measurement_overlay_header_px']={'ensemble':header};write_json(dest/'details.json',row)
    return row


def process_photo(job):
    source,relative,ordinal,output,options=job
    started=time.perf_counter();timings={}
    try:
        original,decoding=read_image(source,return_metadata=True)
        rgb=original if not options['working_width'] else resized(original,options['working_width']);timings['decode_resize']=time.perf_counter()-started
        t=time.perf_counter();calibration,obstruction=calibrate(original,rgb,options);timings['calibration']=time.perf_counter()-t
        settings,physical=settings_for_photo(calibration,[rgb.shape[1],rgb.shape[0]],options)
        calibration['tube_parameters']=physical
        t=time.perf_counter()
        if options.get('mask_dir'):
            mask=Path(options['mask_dir'])/Path(relative).with_suffix('.png')
            labels=read_labels(mask,original.shape,options['mask_format'])
            labels=cv2.resize(labels.astype(np.float32),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int32)
            segmentation={'profile':'imported_mask','automatic_fragment_association':False};links=[]
        else:
            search=resized(rgb,min(rgb.shape[1],960))
            factor=search.shape[1]/rgb.shape[1]
            low=max(3,settings.min_width_px*factor);high=min(60,max(low+1,settings.max_width_px*factor))
            widths=tuple(sorted(set(round(v,1) for v in np.geomspace(low,high,7))))
            labels,segmentation=multiscale_colour_candidates(search,widths)
            segmentation['search_size_xy']=[search.shape[1],search.shape[0]]
            labels=cv2.resize(labels.astype(np.float32),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int32)
            labels,links=associate_labels(labels,settings)
            segmentation['input_normalization']={'original_size_xy':[original.shape[1],original.shape[0]],'analysis_size_xy':[rgb.shape[1],rgb.shape[0]]}
        timings['segmentation_association']=time.perf_counter()-t
        t=time.perf_counter()
        export=write_result
        if options.get('limited_output'):
            from .compact_report import write_result as export
        row=export(output,source,relative,ordinal,original,rgb,decoding,labels,obstruction,calibration,settings,segmentation,links)
        timings['B_refinement_export']=time.perf_counter()-t;timings['total']=time.perf_counter()-started;row['timings_seconds']=timings
        return {'image':row}
    except (OSError,ValueError,KeyError) as exc:
        return {'error':{'photo':relative,'error':str(exc),'elapsed_seconds':time.perf_counter()-started}}


def jobs_for(args,output,reference_calibration=None,reference_size=None):
    options=dict(min_radius_um=args.min_radius_um,max_radius_um=args.max_radius_um,patch_side_um=args.patch_side_um,working_width=args.working_width,scale_mode=args.scale_mode,um_per_pixel=args.um_per_pixel,
                 mask_dir=args.mask_dir,mask_format=args.mask_format,limited_output=getattr(args,'limited_output',False),
                 reference_calibration=reference_calibration,reference_size_xy=reference_size)
    paths=sorted(list_images(args.input,args.recursive),key=lambda p:natural_key(p.relative_to(args.input).as_posix()))
    return [(path,path.relative_to(args.input).as_posix(),i,output,options) for i,path in enumerate(paths,1)]


def execute(jobs):
    for job in jobs:
        print(f'[{job[2]}/{len(jobs)}] {job[1]} · B',flush=True)
        yield process_photo(job)


def run_batch(args,reference_calibration=None,reference_size=None):
    limited=getattr(args,'limited_output',False)
    output=args.output if limited else args.output/'report';jobs=jobs_for(args,output,reference_calibration,reference_size)
    if not jobs:raise ValueError('Нет поддерживаемых фотографий.')
    from .linux_parallel import available_workers,execute as execute_parallel
    workers=available_workers(getattr(args,'workers',1),len(jobs))
    args.effective_workers=workers
    output.mkdir(parents=True);started=time.perf_counter()
    manifest=provenance(args);manifest.update(algorithm='B',experiment='B-continuous-filaments',settings=asdict(TubeSettings()),
          working_coordinates='EXIF-oriented; normalized RGB8 raster, independent of container format; per-image physical settings in calibration.tube_parameters',
          weight_notice='Relative fit scores, not species probabilities',images=[],duplicates=[],errors=[])
    seen={}
    results=execute_parallel(jobs,workers) if workers>1 else execute(jobs)
    for result in results:
        if 'error' in result:manifest['errors'].append(result['error']);print(f"Ошибка: {result['error']}",flush=True);continue
        row=result['image'];digest=row['decoded_rgb_sha256']
        if digest in seen:manifest['duplicates'].append({'photo':row['photo'],'duplicate_of':seen[digest]});continue
        seen[digest]=row['photo'];manifest['images'].append(row)
        print(f"  готово: {row['timings_seconds']['total']:.1f} с",flush=True)
    manifest['images'].sort(key=lambda im:natural_key(im['photo']))
    return finish(output,manifest,started,limited)


def finish(output,manifest,started,limited=False):
    manifest['scale_consistency']=check_scale_consistency([r['calibration'] for r in manifest['images']])
    methods=[p['methods']['ensemble'] for row in manifest['images'] for p in row['proposals']]
    fits=[f for method in methods for f in method['fits']]
    manifest['summary']=dict(unique_images=len(manifest['images']),excluded_duplicate_images=len(manifest['duplicates']),
        common_proposals=len(methods),calibrated_images=sum(r['calibration']['working_um_per_px_xy'] is not None for r in manifest['images']),
        methods={'ensemble':dict(proposal_statuses=dict(Counter(m['status'] for m in methods)),axis_hypotheses=len(fits),
            axes_passing_filter=sum(not f['reasons'] for f in fits),axes_rejected=sum(bool(f['reasons']) for f in fits))},
        notice='Counts are unverified candidates, not biological recall/precision.')
    if limited:
        from .compact_report import build_report as build_compact
        build_compact(output,manifest)
        manifest['total_seconds']=time.perf_counter()-started
        return manifest
    build_report(output,manifest)
    manifest['total_seconds']=time.perf_counter()-started
    write_json(output/'manifest.json',manifest);write_json(output/'summary.json',manifest['summary'])
    diagnostics=['<!doctype html><meta charset="utf-8"><h1>Диагностика B</h1><a href="index.html">Основной отчёт</a>']
    for row in manifest['images']:
        import html
        diagnostics.append(f'<h2>{html.escape(row["photo"])}</h2><img style="max-width:100%" src="images/{row["image_id"]}/diagnostics.jpg">')
    (output/'diagnostics.html').write_text('\n'.join(diagnostics),encoding='utf-8')
    return manifest
