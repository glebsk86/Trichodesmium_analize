"""Physical scale for the tube experiment, with explicit raster transforms."""
import copy
import numpy as np
from .calibration import analyse_axis, ruler_axes, positive, ruler_analysis

FINE_METHOD = 'spectrum_with_individual_tick_support'


def windowed_fine_scale(rgb, tick_um=3., ruler_length_um=600.):
    """Read faint fine marks in short windows; require two agreeing ruler axes.

    Coarse fallback intervals are never relabelled as smallest ticks. Windows
    overlap, so they are diagnostics rather than independent measurements.
    """
    tick_um = positive(tick_um,"tick_um")
    ruler_length_um = positive(ruler_length_um,"ruler_length_um")
    axes = []
    analysis=ruler_analysis(rgb)
    for p0,p1 in ruler_axes(rgb):
        length = np.linalg.norm(p1-p0)
        direction = (p1-p0)/length
        normal = np.array([-direction[1],direction[0]])
        windows = []
        for start in np.arange(0,length-240,120):
            choices = []
            for offset in (-2.,0.,2.):
                a = analyse_axis(rgb,p0+direction*start+normal*offset,
                                 p0+direction*(start+240)+normal*offset,analysis)
                if a and a.get('spacing_method') == FINE_METHOD:
                    a.update(window_start_px=float(start), normal_offset_px=offset)
                    choices.append(a)
            if choices:
                windows.append(max(choices,key=lambda a:a['fundamental_fraction']*len(a['tick_positions_px'])))
        if len(windows)<2:
            continue
        pitch = float(np.median([a['spacing_px'] for a in windows]))
        windows = [a for a in windows if abs(a['spacing_px']/pitch-1)<=.08]
        if len(windows)<2:
            continue
        pitch = float(np.median([a['spacing_px'] for a in windows]))
        axes.append(dict(p0_xy=p0.tolist(),p1_xy=p1.tolist(),spacing_px=pitch,
                         windows=windows,spacing_method=FINE_METHOD,
                         tick_positions_px=sorted(set(round(a['window_start_px']+t,2) for a in windows for t in a['tick_positions_px']))))
    result = dict(method='windowed_fine_ruler',um_per_px=None,quality='не определена',
                  axes=axes,tick_um=tick_um,ruler_length_um=ruler_length_um,
                  reasons=['smallest_ticks_not_resolved_on_two_axes'])
    for i,a in enumerate(axes):
        da = np.subtract(a['p1_xy'],a['p0_xy']);da /= np.linalg.norm(da)
        for b in axes[i+1:]:
            db = np.subtract(b['p1_xy'],b['p0_xy']);db /= np.linalg.norm(db)
            if abs(da@db)<.9 and abs(a['spacing_px']/b['spacing_px']-1)<=.08:
                result.update(um_per_px=tick_um/float(np.mean([a['spacing_px'],b['spacing_px']])),
                              quality='средняя',axes=[a,b],
                              reasons=['experimental_ruler_detection_requires_review',
                                       'overlapping_windows_not_independent',
                                       'full_600_um_endpoints_not_verified'])
                return result
    return result


def image_scale(rgb, seed_calibration, source_to_working_xy):
    """Seed scales with fine-tick evidence are retained; unresolved ones retry."""
    seed = copy.deepcopy(seed_calibration or {})
    value = seed.get('um_per_px')
    fine = seed.get('axes') and all(a.get('spacing_method')==FINE_METHOD for a in seed['axes'])
    explicit = seed.get('method') in ('manual','reference_points','reference')
    if value is not None and np.isfinite(value) and value>0 and (fine or explicit):
        physical = seed
        origin = 'frozen_seed_calibration'
    else:
        physical = windowed_fine_scale(rgb,seed.get('tick_um',3.),seed.get('ruler_length_um',600.))
        origin = 'recomputed_from_source_ruler_windows'
    value = physical['um_per_px']
    factors = np.asarray(source_to_working_xy,dtype=float)
    if factors.shape!=(2,) or not np.isfinite(factors).all() or np.any(factors<=0):
        raise ValueError('Invalid original-to-working scale')
    return dict(source_calibration=physical,seed_calibration=seed,origin=origin,
                source_size_xy=[rgb.shape[1],rgb.shape[0]],
                source_to_working_scale_xy=factors.tolist(),
                working_um_per_px_xy=None if value is None else (value/factors).tolist(),
                notice='Physical scale is experimental and requires visual ruler review.')


def check_scale_consistency(scales, tolerance=.10):
    """Warn only within identical source dimensions; never auto-adjust scale.

    Equal resolution does not prove equal sampling (digital crops may differ).
    This is a review flag, not a reason to discard individually measured scale.
    """
    if not np.isfinite(tolerance) or not 0<tolerance<1:
        raise ValueError('Scale consistency tolerance must lie between 0 and 1')
    groups = {}
    for scale in scales:
        groups.setdefault(tuple(scale['source_size_xy']),[]).append(scale)
    summaries = []
    for size,members in groups.items():
        values = [s['source_calibration']['um_per_px'] for s in members
                  if s['source_calibration']['um_per_px'] is not None]
        reference = float(np.median(values)) if len(values)>=2 else None
        outliers = 0
        for scale in members:
            value = scale['source_calibration']['um_per_px']
            deviation = None if value is None or reference is None else abs(value/reference-1)
            status = ('scale_unavailable' if value is None else 'insufficient_comparison_frames' if reference is None
                      else 'outlier_requires_review' if deviation>tolerance else 'consistent')
            scale['consistency'] = dict(status=status,relative_deviation=deviation,tolerance=tolerance,
                                       median_source_um_per_px=reference,source_size_xy=list(size))
            if status=='outlier_requires_review':
                outliers += 1
                scale['notice'] += ' Scale differs within resolution group: review cropping and ruler.'
        summaries.append(dict(source_size_xy=list(size),comparable_images=len(values),outliers=outliers,
                              median_source_um_per_px=reference))
    return dict(groups=summaries,tolerance=tolerance,
                notice='Compare only equal original dimensions; crops may still differ. No automatic rescaling or conversion suppression.')
