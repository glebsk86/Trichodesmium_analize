"""Visible 2-D geometry; overlapping/branched objects remain unresolved."""
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates, label as connected_components
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


def skeleton_path(mask, prune_spurs=False):
    if connected_components(mask,structure=np.ones((3,3)))[1]>1:
        return np.empty((0,2)),["disconnected_instance_mask"],0,0
    coords = np.argwhere(skeletonize(mask))
    lookup = {tuple(p): i for i, p in enumerate(coords)}
    graph = [[] for _ in coords]
    for i, (y, x) in enumerate(coords):
        for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            j = lookup.get((y+dy, x+dx))
            if j is None:
                continue
            # Do not add a diagonal shortcut around a connected right angle.
            if dy and dx and ((y+dy, x) in lookup or (y, x+dx) in lookup):
                continue
            graph[i].append((j, math.hypot(dy, dx)))
    removed=set()
    if prune_spurs and len(coords):
        # Ignore only short medial-axis twigs caused by ragged proposal edges.
        # Long branches/intersections remain unresolved; do not pick one chain.
        radius=distance_transform_edt(mask)
        limit=min(12.,max(3.,float(np.median(radius[tuple(coords.T)]))*2))
        while True:
            changed=False
            for end in [i for i,g in enumerate(graph) if len(g)==1 and i not in removed]:
                nodes=[end]
                previous=-1
                node=end
                length=0.
                while True:
                    options=[(j,w) for j,w in graph[node] if j!=previous]
                    if not options: break
                    target,weight=options[0]
                    length+=weight
                    previous,node=node,target
                    if len(graph[node])!=2 or length>limit: break
                    nodes.append(node)
                if len(graph[node])>2 and length<=limit:
                    removed.update(nodes)
                    for i in nodes: graph[i]=[]
                    graph[node]=[(j,w) for j,w in graph[node] if j not in removed]
                    changed=True
                    break
            if not changed: break
    ends = [i for i, neighbours in enumerate(graph) if len(neighbours) == 1]
    branch_nodes = sum(len(neighbours) > 2 for neighbours in graph)
    if len(ends) != 2 or branch_nodes:
        return np.empty((0, 2)), ["branched_or_closed_mask"], len(ends), branch_nodes
    start, target = ends
    distances = {start: 0.}
    parent = {}
    queue = [(0., start)]
    while queue:
        dist, i = heapq.heappop(queue)
        if dist != distances[i]:
            continue
        if i == target:
            break
        for j, w in graph[i]:
            candidate = dist+w
            if candidate < distances.get(j, float("inf")):
                distances[j] = candidate
                parent[j] = i
                heapq.heappush(queue, (candidate, j))
    if target not in distances:
        return np.empty((0, 2)), ["disconnected_instance_mask"], len(ends), branch_nodes
    indices = [target]
    while indices[-1] != start:
        indices.append(parent[indices[-1]])
    # Disconnected islands must not silently disappear from one labeled object.
    if len(indices) < len(coords)-len(removed):
        return np.empty((0, 2)), ["disconnected_instance_mask"], len(ends), branch_nodes
    path = coords[indices[::-1]][:, ::-1].astype(float)  # xy
    if len(path) < 5:
        return path, ["short_centerline"], len(ends), branch_nodes
    smooth = gaussian_filter1d(path, 1.3, axis=0, mode="nearest")
    smooth[0], smooth[-1] = path[0], path[-1]
    # Extend skeleton tips to mask boundaries; skeleton endpoints alone undercount.
    distance = distance_transform_edt(mask)
    extension = max(5., float(np.max(distance))*3)
    for end in (0, -1):
        interior = smooth[min(5,len(smooth)-1)] if end == 0 else smooth[max(0,len(smooth)-6)]
        direction = smooth[end]-interior
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction /= norm
        steps = np.arange(.25, extension+.25, .25)
        ray = smooth[end] + steps[:,None]*direction
        inside = map_coordinates(mask.astype(float), [ray[:,1], ray[:,0]], order=1, mode="constant", cval=0) >= .5
        outside = np.flatnonzero(~inside)
        reach = steps[outside[0]]-.125 if len(outside) else 0
        tip = smooth[end] + max(0,reach)*direction
        smooth = np.vstack([tip,smooth]) if end == 0 else np.vstack([smooth,tip])
    return smooth, ["short_skeleton_spurs_ignored"] if removed else [], len(ends), branch_nodes


def arc_samples(path):
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    distances = np.linspace(0, cumulative[-1], max(2,int(cumulative[-1])+1))
    xy = np.column_stack([np.interp(distances,cumulative,path[:,i]) for i in (0,1)])
    tangent = np.column_stack([gaussian_filter1d(xy[:,i],3,order=1,mode="nearest") for i in (0,1)])
    norms = np.linalg.norm(tangent,axis=1)
    tangent /= np.maximum(norms[:,None],1e-9)
    normals = np.column_stack([-tangent[:,1],tangent[:,0]])
    return distances, xy, normals


def measure(mask, rgb, obstruction, prune_spurs=False):
    path, flags, ends, branches = skeleton_path(mask,prune_spurs)
    result = {"flags": flags, "endpoint_count": ends, "branch_nodes": branches,
              "length_px": None, "width_px": None, "width_std_px": None,
              "width_samples_px": [], "cell_length_px": None,
              "cell_intervals_px": [], "cell_count": None, "cell_count_method": "unavailable",
              "path_xy": path.tolist(), "septum_points_xy": [], "width_lines_xy": [],
              "measurement_quality": "низкая", "cell_quality": "не определена"}
    if len(path) < 5 or set(flags)-{"short_skeleton_spurs_ignored"}:
        return result
    s, xy, normals = arc_samples(path)
    result["length_px"] = float(s[-1])
    radii = map_coordinates(distance_transform_edt(mask), [xy[:,1],xy[:,0]],order=1)
    typical_width = float(np.median(radii)*2)
    radius = min(1000.,max(8.,typical_width*2))
    offsets = np.arange(-radius, radius+.125, .25)
    widths, lines = [], []
    edge_margin = max(3.,typical_width)
    for i in range(0,len(s),max(2,len(s)//300)):
        if s[i] < edge_margin or s[-1]-s[i] < edge_margin:
            continue
        ray = xy[i]+offsets[:,None]*normals[i]
        occupied = map_coordinates(mask.astype(float), [ray[:,1],ray[:,0]],order=1,mode="constant",cval=0) >= .5
        center = int(np.argmin(abs(offsets)))
        if not occupied[center]:
            continue
        left = center
        right = center
        while left>0 and occupied[left-1]: left-=1
        while right<len(offsets)-1 and occupied[right+1]: right+=1
        if left==0 or right==len(offsets)-1:
            continue
        blocked = map_coordinates(obstruction.astype(float), [ray[left:right+1,1],ray[left:right+1,0]],order=0,mode="constant",cval=0)
        if np.any(blocked):
            continue
        widths.append(float(offsets[right]-offsets[left]+.25))
        lines.append([ray[left].tolist(),ray[right].tolist()])
    result["width_samples_px"] = widths
    result["width_lines_xy"] = lines
    if widths:
        result["width_px"] = float(np.mean(widths))
        result["width_std_px"] = float(np.std(widths))
        result["measurement_quality"] = "средняя" if len(widths)>=8 else "низкая"
    else:
        result["flags"].append("no_unobstructed_width_samples")
    if np.any(obstruction & mask):
        result["flags"].append("ruler_overlap")
    h,w = mask.shape
    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
        result["flags"].append("frame_truncated_visible_fragment")
    if widths and np.std(widths)/np.mean(widths)>.25:
        result["flags"].append("variable_width_or_segmentation_error")
        result["measurement_quality"] = "низкая"
    # Sample interior intensities across the central part of the filament.
    gray = (rgb[...,0]*.299+rgb[...,1]*.587+rgb[...,2]*.114)/255
    offsets = np.linspace(-typical_width*.22,typical_width*.22,7)
    bands = xy[:,None,:]+normals[:,None,:]*offsets[None,:,None]
    profile = np.mean(map_coordinates(gray,[bands[...,1],bands[...,0]],order=1),axis=1)
    blocked = np.any(map_coordinates(obstruction.astype(float),[bands[...,1],bands[...,0]],order=0),axis=1)>0
    result["intensity_profile"] = profile.tolist()
    result["profile_distance_px"] = s.tolist()
    signal = gaussian_filter1d(profile,.6)-gaussian_filter1d(profile,max(5.,typical_width))
    # Both light and dark septa are possible; choose the polarity with most
    # regular intervals, rather than merging peaks of opposite polarities.
    options = []
    for polarity in (-1,1):
        peaks,_ = find_peaks(signal*polarity,prominence=max(.025,np.std(signal)*1.2),distance=max(3,typical_width*.5))
        peaks = np.array([i for i in peaks if not blocked[i] and s[i]>edge_margin and s[-1]-s[i]>edge_margin],dtype=int)
        pairs = [(a,b) for a,b in zip(peaks[:-1],peaks[1:]) if not blocked[a:b+1].any()]
        intervals = np.array([s[b]-s[a] for a,b in pairs])
        if len(intervals)>=4 and np.std(intervals)/np.mean(intervals)<.25:
            options.append((len(intervals),float(np.std(intervals)/np.mean(intervals)),peaks,intervals,polarity))
    if options:
        _,_,peaks,intervals,polarity = sorted(options,key=lambda a:(-a[0],a[1]))[0]
        cell_length = float(np.mean(intervals))
        result.update(cell_length_px=cell_length,cell_intervals_px=intervals.tolist(),
                      cell_count=int(math.floor(s[-1]/cell_length+.5)),
                      cell_count_method="estimated_from_length_unverified_septa",
                      septum_points_xy=xy[peaks].tolist(),cell_quality="низкая",septum_polarity=polarity)
        result["flags"].append("unverified_septa_and_estimated_cell_count")
    else:
        result["flags"].append("no_regular_septa")
    return result
