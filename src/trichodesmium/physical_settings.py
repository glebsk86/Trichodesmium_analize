"""Physical tube priors and native-raster colour windows per photograph."""
from dataclasses import asdict
import math
import numpy as np
from .tubular import TubeSettings


def settings_for_photo(calibration,size_xy,options):
    factors=calibration.get('working_um_per_px_xy')
    ratio=max(size_xy[0]/960,size_xy[1]/1280)
    minimum=options.get('min_radius_um',1.5);maximum=options.get('max_radius_um',12.)
    patch_um=options.get('patch_side_um',4.8)
    if factors is not None:
        scale=float(np.sqrt(np.prod(factors)))
        density=.8/scale
        low,high=2*minimum/scale,2*maximum/scale
        patch=max(6,math.ceil(patch_um/scale))
        origin='physical_um';notice='Radius bounds are user-configurable trial priors, not species standards.'
    else:
        scale=None;density=max(1.,ratio)
        low,high=4*density,30*density
        patch=max(6,math.ceil(6*density))
        origin='uncalibrated_resolution_prior';notice='No calibration: physical radius bounds were NOT applied; pixel fallback requires review.'
    association=max(3,int(round(5*density)));association+=1-association%2
    settings=TubeSettings(min_width_px=low,max_width_px=high,patch_side_px=patch,
                          association_px=association,min_spine_length_px=60*density,
                          max_gap_px=80*density,pixel_density_factor=density,
                          max_endpoints=48).validate()
    record=dict(origin=origin,working_um_per_px=scale,
                requested_radius_bounds_um=[minimum,maximum],
                effective_radius_bounds_px=[low/2,high/2],
                effective_radius_bounds_um=None if scale is None else [minimum,maximum],
                patch_side_px=patch,patch_side_um=None if scale is None else patch*scale,
                analysis_size_xy=list(size_xy),settings=asdict(settings),notice=notice)
    return settings,record
