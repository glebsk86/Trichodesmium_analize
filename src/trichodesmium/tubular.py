"""Experimental tube priors. Templates are hypotheses, not measured tissue.

A: one smooth spine and a width prior, refined using pigment contrast.
B: multiple centre/width hypotheses, ranked by two-sided square-patch Lab
gradients. The same source proposals and extracted spines are used by both.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter1d, map_coordinates
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


@dataclass(frozen=True)
class TubeSettings:
    min_width_px: float = 4.0  # Trial limit at WORKING resolution, not a species standard.
    max_width_px: float = 30.0
    patch_side_px: int = 2
    association_px: int = 5
    min_spine_length_px: float = 60.0
    max_local_turn_deg: float = 65.0
    max_endpoints: int = 24
    max_gap_px: float = 35.0
    min_supported_fraction: float = .50
    min_proposal_coverage: float = .55
    min_pigment_difference: float = .6
    min_lab_gradient: float = .7
    max_width_relative_mad: float = .30
    max_curvature_width: float = 1.0
    crossing_angle_deg: float = 25.0
    smoothness_weight: float = 3.0

    def validate(self):
        if not 0 < self.min_width_px < self.max_width_px:
            raise ValueError("Require 0 < min-width < max-width")
        if not 1 < self.patch_side_px < self.min_width_px:
            raise ValueError("Square patch must span multiple pixels and be smaller than min-width")
        if not np.isfinite(self.smoothness_weight) or self.smoothness_weight < 0:
            raise ValueError("Smoothness weight must be finite and nonnegative")
        return self


def sample(array, xy, order=1):
    if array.ndim == 2:
        return map_coordinates(array.astype(np.float32), [xy[..., 1], xy[..., 0]],
                               order=order, mode="constant", cval=0)
    return np.stack([sample(array[..., c], xy, order) for c in range(array.shape[-1])], axis=-1)


def resample_spine(xy, step=2.):
    if len(xy) < 2:
        return xy
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    keep = np.r_[True, np.diff(s) > 1e-6]
    s, xy = s[keep], xy[keep]
    if len(s) < 2:
        return xy
    targets = np.linspace(0, s[-1], max(3, int(s[-1]/step)+1))
    return np.column_stack([np.interp(targets, s, xy[:, c]) for c in range(2)])


def smooth_spine(xy):
    xy = resample_spine(xy)
    if len(xy) < 5:
        return xy
    result = gaussian_filter1d(xy, 1.5, axis=0, mode="nearest")
    result[0], result[-1] = xy[0], xy[-1]
    return resample_spine(result)


def normals(xy):
    tangent = np.gradient(xy, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1e-6)
    return np.c_[-tangent[:, 1], tangent[:, 0]]


def local_turn(xy):
    p = resample_spine(xy, 5.)
    if len(p) < 5:
        return 0.
    a, b = p[2:-2]-p[:-4], p[4:]-p[2:-2]
    a /= np.maximum(np.linalg.norm(a, axis=1)[:, None], 1e-6)
    b /= np.maximum(np.linalg.norm(b, axis=1)[:, None], 1e-6)
    return float(np.max(np.degrees(np.arccos(np.clip(np.sum(a*b, axis=1), -1, 1)))))


def curvature_width(xy, width):
    """Robust bend severity over 10-pixel spans; reject tight folded blobs."""
    p = resample_spine(xy, 5.)
    if len(p) < 5:
        return 0.
    a, b = p[2:-2]-p[:-4], p[4:]-p[2:-2]
    la, lb = np.linalg.norm(a,axis=1), np.linalg.norm(b,axis=1)
    cosine = np.sum(a*b,axis=1)/np.maximum(la*lb,1e-6)
    curvature = np.arccos(np.clip(cosine,-1,1))/np.maximum((la+lb)/2,1)
    return float(np.percentile(curvature,95)*width)


def boundary_regularity(xy, left, right, reference_width_px=None):
    """Reward straight sides and wide smooth bends at a width-relative scale.

    Bending cost measures width/radius squared; waviness cost measures how
    rapidly signed curvature changes. Round tip caps are intentionally absent.
    These are shape priors, never evidence of biological identity.
    """
    width = max(1., float(np.median(left+right)) if reference_width_px is None else float(reference_width_px))
    normal = normals(xy)
    bending, waviness = [], []
    for edge in (xy-left[:,None]*normal, xy+right[:,None]*normal):
        p = resample_spine(edge, max(2., width/4))
        if len(p) < 7:
            continue
        a, b = p[2:-2]-p[:-4], p[4:]-p[2:-2]
        la, lb = np.linalg.norm(a,axis=1), np.linalg.norm(b,axis=1)
        angle = np.arctan2(a[:,0]*b[:,1]-a[:,1]*b[:,0],np.sum(a*b,axis=1))
        curvature = angle/np.maximum((la+lb)/2,1e-6)
        bending.append(float(np.mean(np.minimum(abs(curvature)*width,3.)**2)))
        ds = np.maximum(np.linalg.norm(np.diff(p[2:-2],axis=0),axis=1),1e-6)
        change = np.diff(curvature)/ds*width**2
        waviness.append(float(np.mean(np.minimum(abs(change),3.)**2)))
    bend_cost = float(np.mean(bending)) if bending else 0.
    wave_cost = float(np.mean(waviness)) if waviness else 0.
    regularity = float(np.exp(-bend_cost-.30*wave_cost))
    return {"boundary_regularity":regularity,"boundary_bending_cost":bend_cost,
            "boundary_waviness_cost":wave_cost,"regularity_reference_width_px":width}


def skeleton_graph(mask, spur_limit):
    coords = np.argwhere(skeletonize(mask))
    lookup = {tuple(p): i for i, p in enumerate(coords)}
    graph = [set() for _ in coords]
    for i, (y, x) in enumerate(coords):
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            j = lookup.get((y+dy, x+dx))
            if j is None or (dy and dx and ((y+dy,x) in lookup or (y,x+dx) in lookup)):
                continue
            graph[i].add(j)
    changed = True
    while changed:
        changed = False
        for end in [i for i, g in enumerate(graph) if len(g) == 1]:
            chain, previous, node, length = [end], -1, end, 0.
            while True:
                onward = graph[node]-{previous}
                if not onward:
                    break
                target = next(iter(onward))
                length += np.linalg.norm(coords[target]-coords[node])
                previous, node = node, target
                if len(graph[node]) != 2 or length > spur_limit:
                    break
                chain.append(node)
            if len(graph[node]) > 2 and length <= spur_limit:
                removed = set(chain)
                for i in chain:
                    for j in graph[i]:
                        graph[j].discard(i)
                    graph[i].clear()
                changed = True
                break
    edges = [(i,j,float(np.linalg.norm(coords[i]-coords[j]))) for i,g in enumerate(graph) for j in g]
    if not edges:
        return coords, graph, csr_matrix((len(coords),len(coords)))
    a,b,w = zip(*edges)
    return coords, graph, csr_matrix((w,(a,b)), shape=(len(coords),len(coords)))


def join_fragments(paths, settings):
    """Join only tangent-compatible nearby hypotheses, never measurement pixels."""
    paths = list(paths)
    gaps = []
    while True:
        options = []
        for i in range(len(paths)):
            for j in range(i+1,len(paths)):
                for ri in (False,True):
                    for rj in (False,True):
                        a = paths[i][::-1] if ri else paths[i]
                        b = paths[j][::-1] if rj else paths[j]
                        if min(len(a),len(b)) < 6:
                            continue
                        gap = b[0]-a[-1];distance = np.linalg.norm(gap)
                        if not 2 < distance <= settings.max_gap_px:
                            continue
                        direction = gap/distance
                        u, v = a[-1]-a[-6], b[5]-b[0]
                        u /= np.linalg.norm(u);v /= np.linalg.norm(v)
                        if min(np.dot(u,direction),np.dot(v,direction),np.dot(u,v)) < .87:
                            continue
                        options.append((distance,i,j,ri,rj))
        if not options:
            break
        _,i,j,ri,rj = min(options)
        a = paths[i][::-1] if ri else paths[i]
        b = paths[j][::-1] if rj else paths[j]
        gaps.append([a[-1].tolist(),b[0].tolist()])
        bridge = np.linspace(a[-1],b[0],max(3,int(np.linalg.norm(a[-1]-b[0])/2)))[1:-1]
        joined = smooth_spine(np.vstack([a,bridge,b]))
        paths = [p for k,p in enumerate(paths) if k not in (i,j)]+[joined]
    return paths,gaps


def extract_spines(source, settings):
    association = cv2.morphologyEx(source.astype(np.uint8),cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (settings.association_px,settings.association_px))) > 0
    association |= source
    association = binary_fill_holes(association)  # Axis construction ONLY.
    paths, cases = [], []
    for region in regionprops(label(association,connectivity=2)):
        if region.area < 80:
            continue
        mask = np.pad(region.image,1)
        sk = skeletonize(mask)
        radius = float(np.median(distance_transform_edt(mask)[sk]))
        coords,graph,matrix = skeleton_graph(mask,min(20.,max(6.,radius*2.5)))
        ends = [i for i,g in enumerate(graph) if len(g)==1]
        if len(ends) < 2:
            cases.append("closed_or_no_resolvable_axis")
            continue
        if len(ends) > settings.max_endpoints:
            cases.append("complex_junction_network")
            continue
        distances,pred = dijkstra(matrix,indices=ends,return_predecessors=True)
        candidates = []
        offset = np.array([region.bbox[1]-1,region.bbox[0]-1])
        for a in range(len(ends)):
            for b in range(a+1,len(ends)):
                length = distances[a,ends[b]]
                if not np.isfinite(length) or length < settings.min_spine_length_px:
                    continue
                nodes = [ends[b]]
                while nodes[-1] != ends[a]:
                    predecessor = pred[a,nodes[-1]]
                    if predecessor < 0:break
                    nodes.append(predecessor)
                if nodes[-1] != ends[a]:continue
                xy = smooth_spine(coords[nodes[::-1]][:,::-1]+offset)
                turn = local_turn(xy)
                if turn > settings.max_local_turn_deg:
                    continue
                candidates.append((float(length)/(1+turn/35),xy))
        occupied = np.zeros(source.shape,np.uint8)
        for _,xy in sorted(candidates,key=lambda p:p[0],reverse=True):
            novel = sample(occupied,xy,0)==0
            if np.mean(novel) < .30 or novel.sum()*2 < settings.min_spine_length_px:
                continue
            paths.append(xy)
            cv2.polylines(occupied,[np.rint(xy).astype(np.int32)],False,1,
                          max(3,int(radius*2)))
        if not candidates:
            cases.append("no_sufficiently_smooth_long_axis")
    paths,gaps = join_fragments(paths,settings)
    if gaps:cases.append("gap_bridged_for_template_only")
    return paths,sorted(set(cases)),gaps


class BoundaryEvidence:
    def __init__(self,rgb,field,settings):
        self.lab = cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB).astype(np.float32)
        self.pigment = self.lab[...,2]-.7*self.lab[...,1]
        self.field = field
        self.settings = settings
        k = settings.patch_side_px
        self.patch_lab = cv2.boxFilter(self.lab,-1,(k,k),normalize=True)
        # Even box kernels have their centroid half a pixel before the anchor.
        self.box_shift = .5 if k%2==0 else 0.
        self.offset = math.sqrt(2)*(k-1)/2+.65

    def edge_scores(self,xy,norm,radii,side,kind):
        centres = xy[:,None,:]+side*radii[None,:,None]*norm[:,None,:]
        inside = centres-side*self.offset*norm[:,None,:]
        outside = centres+side*self.offset*norm[:,None,:]
        if kind == "single":
            score = sample(self.pigment,inside)-sample(self.pigment,outside)
        else:
            a = sample(self.patch_lab,inside+self.box_shift)
            b = sample(self.patch_lab,outside+self.box_shift)
            delta = a-b
            gradient = np.linalg.norm(delta,axis=-1)/(2*self.offset)
            signed_pigment = delta[...,2]-.7*delta[...,1]
            score = .55*gradient+.45*signed_pigment/(2*self.offset)
        valid = (sample(self.field,inside,0)>.5)&(sample(self.field,outside,0)>.5)
        return np.where(valid,score,-1e3)


def choose_edges(evidence,xy,radius,kind):
    n = normals(xy)
    s = evidence.settings
    low = max(s.min_width_px/2,radius*.65)
    high = min(s.max_width_px/2,radius*1.35)
    radii = np.linspace(low,max(low+.1,high),17)
    chosen,raw,scores = [],[],[]
    for side in (-1,1):
        image_score = evidence.edge_scores(xy,n,radii,side,kind)
        # Prefer nearby widths softly, not a forced constant radius.
        penalty = 1.0*((radii-radius)/max(radius,1))**2
        indices = np.argmax(image_score-penalty[None,:],axis=1)
        edge = radii[indices]
        raw.append(edge)
        smooth = gaussian_filter1d(edge,2.5,mode="nearest")
        chosen.append(smooth)
        # Re-evaluate the final smooth boundary, not just its local maxima.
        positions = xy+side*smooth[:,None]*n
        inside = positions-side*evidence.offset*n
        outside = positions+side*evidence.offset*n
        if kind=="single":
            score = sample(evidence.pigment,inside)-sample(evidence.pigment,outside)
        else:
            delta = sample(evidence.patch_lab,inside+evidence.box_shift)-sample(evidence.patch_lab,outside+evidence.box_shift)
            score = .55*np.linalg.norm(delta,axis=-1)/(2*evidence.offset)+.45*(delta[:,2]-.7*delta[:,1])/(2*evidence.offset)
        valid = (sample(evidence.field,inside,0)>.5)&(sample(evidence.field,outside,0)>.5)
        scores.append(np.where(valid,score,-1e3))
    return chosen,raw,scores


def render_tube(shape,xy,left,right,valid=None):
    n = normals(xy)
    mask = np.zeros(shape,np.uint8)
    for i in range(len(xy)-1):
        if valid is not None and not (valid[i] and valid[i+1]):continue
        polygon = np.array([xy[i]-left[i]*n[i],xy[i+1]-left[i+1]*n[i+1],
                            xy[i+1]+right[i+1]*n[i+1],xy[i]+right[i]*n[i]])
        cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
    # Circular caps only at actual hypothesis tips; interior gaps have no caps.
    for i in (0,len(xy)-1):
        if valid is None or valid[i]:
            cv2.circle(mask,tuple(np.rint(xy[i]).astype(int)),max(1,int(round((left[i]+right[i])/2))),1,-1)
    return mask>0


def fit_hypothesis(evidence,source,xy,initial_radius,obstruction,kind,shift=0.,factor=1.):
    s = evidence.settings
    shifted = xy+shift*normals(xy)
    radius = float(np.clip(initial_radius*factor,s.min_width_px/2,s.max_width_px/2))
    chosen,raw,scores = choose_edges(evidence,shifted,radius,kind)
    widths = raw[0]+raw[1]
    median = float(np.median(widths))
    width_mad = float(np.median(abs(widths-median))/max(median,1))
    template = render_tube(source.shape,shifted,*chosen)
    threshold = s.min_pigment_difference if kind=="single" else s.min_lab_gradient
    border_supported = np.minimum(scores[0],scores[1])>=threshold
    blocked = sample(obstruction,shifted,0)>.5
    observed = border_supported&~blocked
    near_source = sample(distance_transform_edt(~source),shifted)<=max(3.,radius)
    # Reject a contrast fit which slid away from the common proposal.
    observed &= near_source
    supported = render_tube(source.shape,shifted,*chosen,valid=observed)&~obstruction&evidence.field
    supported &= cv2.dilate(source.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
    outside = ~blocked
    fraction = float(np.mean(observed[outside])) if outside.any() else 0.
    gradient = float(np.mean(np.clip(np.minimum(scores[0],scores[1]),-5,20)[outside])) if outside.any() else -5.
    geometric_length = float(np.linalg.norm(np.diff(shifted,axis=0),axis=1).sum())
    # Objective weights are relative fit scores, never identity probabilities.
    base_objective = gradient+2*fraction-width_mad*3-abs(shift)/max(initial_radius,1)*.3
    # All seven variants share one reference width: merely shrinking the mask
    # must not win extra points through its normalization denominator.
    shape = boundary_regularity(shifted,*chosen,reference_width_px=2*initial_radius)
    smoothness_bonus = s.smoothness_weight*shape["boundary_regularity"] if kind=="ensemble" else 0.
    objective = base_objective+smoothness_bonus
    reasons=[]
    if fraction<s.min_supported_fraction:reasons.append("insufficient_bilateral_boundary_support")
    if width_mad>s.max_width_relative_mad:reasons.append("observed_width_inconsistent")
    if geometric_length/max(median,1)<6:reasons.append("too_short_for_width")
    bend = curvature_width(shifted,median)
    if bend>s.max_curvature_width:reasons.append("bend_too_tight_for_width")
    return {"template":template,"supported":supported,"axis_xy":shifted,
            "left_radius_px":chosen[0],"right_radius_px":chosen[1],"supported_sections":observed,
            "scores":scores,"objective":objective,"supported_fraction":fraction,
            "median_proposed_width_px":median,"raw_width_relative_mad":width_mad,
            "template_axis_length_px":geometric_length,"reasons":reasons,
            "curvature_width_95":bend,
            "base_objective":base_objective,"smoothness_bonus":smoothness_bonus,**shape,
            "shift_px":shift,"radius_factor":factor}


def fit_spine(evidence,source,xy,obstruction,kind):
    solid = binary_fill_holes(cv2.morphologyEx(source.astype(np.uint8),cv2.MORPH_CLOSE,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))>0)
    distances = distance_transform_edt(np.pad(solid,1))[1:-1,1:-1]
    estimates = sample(distances,xy)
    estimates = estimates[estimates>=evidence.settings.min_width_px/2]
    radius = float(np.median(estimates)) if len(estimates) else evidence.settings.min_width_px/2
    radius = float(np.clip(radius,evidence.settings.min_width_px/2,evidence.settings.max_width_px/2))
    if kind=="single":
        variants=[fit_hypothesis(evidence,source,xy,radius,obstruction,kind)]
    else:
        variants=[fit_hypothesis(evidence,source,xy,radius,obstruction,kind,shift,factor)
                  for shift,factor in ((0.,.75),(0.,.9),(0.,1.),(0.,1.15),(0.,1.3),
                                       (-radius*.25,1.),(radius*.25,1.))]
    objective=np.array([v["objective"] for v in variants])
    weights=np.exp(np.clip(objective-objective.max(),-30,0));weights/=weights.sum()
    for v,w in zip(variants,weights):v["relative_fit_weight"]=float(w)
    eligible=[i for i,v in enumerate(variants) if not v["reasons"]]
    winner=max(eligible,key=lambda i:objective[i]) if eligible else int(np.argmax(objective))
    result=dict(variants[winner])
    result["selected_variant"]=winner
    result["variants"]=variants if kind=="ensemble" else []
    return result


def pair_overlap(first,second,settings):
    """Distinguish angular crossings from collinear/shared-axis ambiguity."""
    overlap=first["template"]&second["template"]
    if overlap.sum()<5:
        return overlap, None, None
    yx=np.argwhere(overlap)
    target=np.median(yx[:,::-1],axis=0)
    directions=[]
    near_tip=[]
    for fit in (first,second):
        xy=fit["axis_xy"]
        i=int(np.argmin(np.linalg.norm(xy-target,axis=1)))
        widths=fit["left_radius_px"]+fit["right_radius_px"]
        near_tip.append(min(np.linalg.norm(xy[i]-xy[0]),np.linalg.norm(xy[i]-xy[-1]))<float(np.median(widths)))
        a,b=max(0,i-5),min(len(xy)-1,i+5)
        direction=xy[b]-xy[a]
        directions.append(direction/max(np.linalg.norm(direction),1e-6))
    angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(*directions)),0,1))))
    kind=("endpoint_junction" if angle>=settings.crossing_angle_deg and any(near_tip) else
          "angular_crossing" if angle>=settings.crossing_angle_deg else "shared_axis_or_touching")
    return overlap,kind,angle


def compare_proposal(rgb,source,field,obstruction,settings=TubeSettings()):
    settings.validate()
    paths,axis_cases,gaps=extract_spines(source,settings)
    evidence=BoundaryEvidence(rgb,field,settings)
    results={}
    for kind in ("single","ensemble"):
        fits=[fit_spine(evidence,source,xy,obstruction,kind) for xy in paths]
        union=np.zeros(source.shape,bool)
        active=[f for f in fits if not f["reasons"]]
        for fit in active:union|=fit["template"]
        source_coverage=float(np.mean(union[source])) if source.any() else 0.
        crossing=np.zeros(source.shape,bool)
        ambiguous=np.zeros(source.shape,bool)
        pairs=[]
        for i in range(len(fits)):
            for j in range(i+1,len(fits)):
                if fits[i]["reasons"] or fits[j]["reasons"]:continue
                overlap,overlap_kind,angle=pair_overlap(fits[i],fits[j],settings)
                if overlap_kind:
                    pairs.append({"spines":[i+1,j+1],"kind":overlap_kind,"angle_deg":angle})
                    ambiguous|=overlap
                    if overlap_kind=="angular_crossing":crossing|=overlap
        reasons=[]
        if not fits:reasons.append("no_tube_axis")
        if source_coverage<settings.min_proposal_coverage:reasons.append("proposal_not_explained_by_tubes")
        if fits and all(f["reasons"] for f in fits):reasons.append("all_spines_failed_boundary_checks")
        is_crossing=bool(crossing.any()) or "complex_junction_network" in axis_cases
        status=("crossing_requires_review" if is_crossing else
                "junction_requires_review" if any(p["kind"]=="endpoint_junction" for p in pairs) else
                "shared_axis_requires_review" if ambiguous.any() else
                "partly_explained_requires_review" if active and reasons else
                "rejected" if reasons else "accepted_candidate")
        # Shared junction pixels have no unique ownership. Keep hypothesis masks
        # individually but exclude intersections from evidence-based measurements.
        for fit in fits:
            fit["status"]="rejected" if fit["reasons"] else "accepted_candidate"
            fit["supported"] &= ~ambiguous
        results[kind]={"fits":fits,"source_coverage":source_coverage,"status":status,
                       "reasons":reasons,"crossing_mask":crossing,
                       "ambiguous_mask":ambiguous,"overlap_pairs":pairs}
    return {"axis_cases":axis_cases,"template_gap_links_xy":gaps,"methods":results,
            "settings":asdict(settings)}
