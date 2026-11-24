"""Input-only candidate stable-region selection and established ICP controls.

This module does not read dataset metadata, known transformations or truth labels.
The worker entry point can deny access to every dataset file except inputs.npz.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

METHODS = ("ordinary", "trimmed_90", "huber", "normal_space", "pointwise_gate", "patch_no_dilation", "patch_candidate")


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def fit_rigid(source, target, weights=None):
    weights = np.ones(len(source)) if weights is None else weights
    weights = weights / weights.sum()
    a, b = weights @ source, weights @ target
    u, _, vt = np.linalg.svd(((source-a)*weights[:, None]).T @ (target-b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, b-rotation @ a
    return matrix


def icp(reference, moving, config, *, mode="ordinary", initial=None, iterations=None, spacing=None):
    tree = cKDTree(reference)
    matrix = np.eye(4) if initial is None else initial.copy()
    cloud = transform(moving, matrix)
    previous, history = None, []
    limit = config["max_iterations"] if iterations is None else iterations
    for iteration in range(limit):
        distances, index = tree.query(cloud, k=1)
        keep, weights = np.arange(len(cloud)), None
        if mode == "trimmed":
            count = max(3, int(np.floor(len(cloud)*config["trimmed_keep_fraction"])))
            keep = np.sort(np.argpartition(distances, count-1)[:count])
        if mode == "huber":
            scale = max(1.4826*np.median(np.abs(distances-np.median(distances))), .05*spacing)
            cutoff = config["huber_multiplier"]*scale
            weights = np.minimum(1.0, cutoff/np.maximum(distances, 1e-15))
        delta = fit_rigid(cloud[keep], reference[index[keep]], weights)
        cloud = transform(cloud, delta)
        matrix = delta @ matrix
        updated = tree.query(cloud)[0]
        rmse = float(np.sqrt(np.mean(updated**2)))
        history.append({"iteration":iteration+1, "nn_rmse_m":rmse, "used_pairs":len(keep),
                        "effective_weight_sum":float(weights.sum()) if weights is not None else float(len(keep))})
        if previous is not None and abs(previous-rmse) < config["tolerance_rmse_m"]:
            break
        previous = rmse
    return matrix, history


def estimate_normals(points, k):
    _, index = cKDTree(points).query(points, k=min(k, len(points)))
    neighborhoods = points[index]
    centered = neighborhoods-neighborhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)/centered.shape[1]
    _, vectors = np.linalg.eigh(covariance)
    normals = vectors[:, :, 0]
    # Height-field terrain convention: orient estimated normals towards +Z.
    normals[normals[:, 2] < 0] *= -1
    return normals


def normal_sample(normals, budget, config, seed):
    phi = (np.arctan2(normals[:, 1], normals[:, 0])+np.pi)/(2*np.pi)
    az = np.minimum((phi*config["normal_azimuth_bins"]).astype(int), config["normal_azimuth_bins"]-1)
    z = np.minimum((np.clip(normals[:,2],0,1)*config["normal_cosine_bins"]).astype(int), config["normal_cosine_bins"]-1)
    labels = az*config["normal_cosine_bins"]+z
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(np.flatnonzero(labels==label)).tolist() for label in np.unique(labels)]
    selected = []
    while len(selected) < min(budget, len(normals)):
        for bucket in buckets:
            if bucket:
                selected.append(bucket.pop())
                if len(selected)==min(budget,len(normals)):
                    break
    return np.sort(np.asarray(selected, dtype=int))


def connected_cores(flagged, minimum):
    remaining, retained = set(flagged), set()
    neighbors = [offset for offset in product((-1,0,1), repeat=3) if offset != (0,0,0)]
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, component = [first], [first]
        while stack:
            current = stack.pop()
            for offset in neighbors:
                candidate = tuple(current[i]+offset[i] for i in range(3))
                if candidate in remaining and flagged[candidate]==flagged[first]:
                    remaining.remove(candidate)
                    stack.append(candidate)
                    component.append(candidate)
        if len(component) >= minimum:
            retained.update(component)
    return retained


def dilate(cells, radius):
    return {tuple(cell[i]+offset[i] for i in range(3)) for cell in cells
            for offset in product(range(-radius,radius+1),repeat=3)}


def geometry_evidence(reference, moving, coarse, normals, spacing, config):
    aligned = transform(moving, coarse)
    forward_distance, forward_index = cKDTree(reference).query(aligned)
    reverse_distance, reverse_index = cKDTree(aligned).query(reference)
    forward_r = np.einsum("ij,ij->i", aligned-reference[forward_index], normals[forward_index])
    reverse_r = np.einsum("ij,ij->i", aligned[reverse_index]-reference, normals)
    maximum_distance = config["maximum_match_distance_spacing"]*spacing
    valid_m, valid_r = forward_distance <= maximum_distance, reverse_distance <= maximum_distance
    residuals = np.r_[forward_r[valid_m],reverse_r[valid_r]]
    location = float(np.median(residuals))
    sigma = float(max(1.4826*np.median(abs(residuals-location)), config["residual_sigma_floor_spacing"]*spacing))
    threshold = config["residual_sigma_multiplier"]*sigma
    patch_size = config["patch_spacing_multiplier"]*spacing
    origin = np.floor(reference.min(0)/patch_size)*patch_size
    keys_r = np.floor((reference-origin)/patch_size).astype(int)
    # Anchor both directions to reference geometry, so vertical changes do not split a patch into unrelated cells.
    keys_m = keys_r[forward_index]
    pool_keys = np.vstack((keys_m[valid_m],keys_r[valid_r]))
    pool_r = residuals-location
    cells, inverse = np.unique(pool_keys,axis=0,return_inverse=True)
    order = np.argsort(inverse,kind="stable")
    boundaries = np.r_[0,np.flatnonzero(np.diff(inverse[order]))+1,len(order)]
    flagged = {}
    statistics = []
    for start, end in zip(boundaries[:-1],boundaries[1:]):
        label = inverse[order[start]]
        values = pool_r[order[start:end]]
        median = float(np.median(values))
        fraction = float(np.mean(values*np.sign(median) > threshold/2)) if median else 0.0
        is_flagged = len(values)>=config["minimum_patch_pairs"] and abs(median)>threshold and fraction>=config["sign_consistency_fraction"]
        key = tuple(cells[label].tolist())
        if is_flagged:
            flagged[key] = int(np.sign(median))
        statistics.append({"cell_x":key[0],"cell_y":key[1],"cell_z":key[2],"paired_observations":len(values),
                           "median_centered_normal_residual_m":median,"same_sign_exceedance_fraction":fraction,"flagged":bool(is_flagged)})
    cores = connected_cores(flagged,config["minimum_connected_cells"])
    return {"residual_m":forward_r,"residual_r":reverse_r,"valid_m":valid_m,"valid_r":valid_r,
            "keys_m":keys_m,"keys_r":keys_r,"location":location,"sigma":sigma,"threshold":threshold,
            "patch_size":patch_size,"cores":cores,"patch_statistics":statistics,
            "forward_distance":forward_distance,"reverse_distance":reverse_distance}


def mask_from_cells(keys, cells):
    return np.fromiter((tuple(key) not in cells for key in keys),dtype=bool,count=len(keys))


def run_methods(reference, moving, config):
    started = time.perf_counter()
    spacing = float(np.median(cKDTree(reference).query(reference,k=2)[0][:,1]))
    reference_normals = estimate_normals(reference,config["normal_neighbors"])
    reference_feature_time = time.perf_counter()-started
    outputs = {}
    ref_all, mov_all = np.ones(len(reference),bool), np.ones(len(moving),bool)
    for method, mode in (("ordinary","ordinary"),("trimmed_90","trimmed"),("huber","huber")):
        started=time.perf_counter()
        matrix,history=icp(reference,moving,config,mode=mode,spacing=spacing)
        elapsed=time.perf_counter()-started
        outputs[method]={"matrix":matrix,"reference_mask":ref_all.copy(),"moving_mask":mov_all.copy(),"history":history,
                         "elapsed_seconds":elapsed,"fit_iterations":len(history),"selection_info":{"mode":mode,"fallback":False}}
    started=time.perf_counter()
    moving_normals=estimate_normals(moving,config["normal_neighbors"])
    ref_idx=normal_sample(reference_normals,config["normal_sample_budget"],config,config["random_seed"])
    mov_idx=normal_sample(moving_normals,config["normal_sample_budget"],config,config["random_seed"]+1)
    matrix,history=icp(reference[ref_idx],moving[mov_idx],config)
    ref_mask,mov_mask=np.zeros(len(reference),bool),np.zeros(len(moving),bool)
    ref_mask[ref_idx],mov_mask[mov_idx]=True,True
    outputs["normal_space"]={"matrix":matrix,"reference_mask":ref_mask,"moving_mask":mov_mask,"history":history,
        "elapsed_seconds":reference_feature_time+time.perf_counter()-started,"fit_iterations":len(history),
        "selection_info":{"fallback":False,"mode":"hemisphere normal-space stratified sample"}}
    started=time.perf_counter()
    coarse,coarse_history=icp(reference,moving,config,iterations=config["coarse_iterations"])
    coarse_time=time.perf_counter()-started
    started=time.perf_counter()
    evidence=geometry_evidence(reference,moving,coarse,reference_normals,spacing,config)
    evidence_time=time.perf_counter()-started
    for method in ("pointwise_gate","patch_no_dilation","patch_candidate"):
        started=time.perf_counter()
        if method=="pointwise_gate":
            ref_mask=evidence["valid_r"] & (abs(evidence["residual_r"]-evidence["location"])<=evidence["threshold"])
            mov_mask=evidence["valid_m"] & (abs(evidence["residual_m"]-evidence["location"])<=evidence["threshold"])
            excluded_cells=set()
        else:
            excluded_cells=dilate(evidence["cores"],config["dilation_cells"]) if method=="patch_candidate" else evidence["cores"]
            ref_mask=evidence["valid_r"] & mask_from_cells(evidence["keys_r"],excluded_cells)
            mov_mask=evidence["valid_m"] & mask_from_cells(evidence["keys_m"],excluded_cells)
        fallback=min(ref_mask.mean(),mov_mask.mean()) < config["minimum_retained_fraction"] or min(ref_mask.sum(),mov_mask.sum())<30
        info={"mode":method,"fallback":bool(fallback),"excluded_core_cells":len(evidence["cores"]),
              "excluded_cells_after_expansion":len(excluded_cells),"normal_residual_location_m":evidence["location"],
              "normal_residual_sigma_m":evidence["sigma"],"normal_residual_threshold_m":evidence["threshold"],
              "patch_size_m":evidence["patch_size"]}
        if fallback:
            ref_mask,mov_mask=ref_all.copy(),mov_all.copy()
            matrix,history=coarse.copy(),[]
            info["fallback_reason"]="Too few retained points; return coarse ordinary result without reading any labels"
        else:
            matrix,history=icp(reference[ref_mask],moving[mov_mask],config,initial=coarse,iterations=config["refine_iterations"])
        outputs[method]={"matrix":matrix,"reference_mask":ref_mask,"moving_mask":mov_mask,
            "history":[dict(row,stage="coarse") for row in coarse_history]+[dict(row,stage="refine") for row in history],
            "elapsed_seconds":reference_feature_time+coarse_time+evidence_time+time.perf_counter()-started,
            "fit_iterations":len(coarse_history)+len(history),"selection_info":info}
    diagnostics={"spacing_m":spacing,"reference_normal_estimation_and_spacing_seconds":reference_feature_time,
                 "normal_residual_location_m":evidence["location"],"normal_residual_sigma_m":evidence["sigma"],
                 "normal_residual_threshold_m":evidence["threshold"],"patch_size_m":evidence["patch_size"],
                 "core_change_cells":len(evidence["cores"]),"coarse_iterations":len(coarse_history),
                 "normal_orientation":"+Z, assumes terrain-like height-field geometry",
                 "timing":"Each feature-using method is charged common preprocessing time even if it was cached; excludes file exports"}
    return outputs,diagnostics,evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config",required=True)
    parser.add_argument("--dataset-root",required=True)
    args=parser.parse_args()
    input_path,output_path=Path(args.input).resolve(),Path(args.output).resolve()
    dataset_root=Path(args.dataset_root).resolve()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_path.mkdir(parents=True,exist_ok=True)
    access=[]
    def audit(event,arguments):
        if event!="open" or not arguments or not isinstance(arguments[0],(str,bytes,Path)):
            return
        try:
            path=Path(arguments[0]).resolve()
        except (OSError,ValueError,TypeError):
            return
        if path==dataset_root or dataset_root in path.parents:
            allowed=path==input_path
            access.append({"path":str(path),"allowed":allowed})
            if not allowed:
                raise PermissionError(f"Input-only worker cannot read other dataset files: {path}")
    sys.addaudithook(audit)
    with np.load(input_path) as saved:
        if set(saved.files)!={"reference","moving"}:
            raise ValueError("Algorithm input must contain only reference and moving coordinates")
        reference,moving=saved["reference"].copy(),saved["moving"].copy()
    if not np.isfinite(reference).all() or not np.isfinite(moving).all():
        raise ValueError("Nonfinite coordinates")
    outputs,diagnostics,evidence=run_methods(reference,moving,config)
    for method in METHODS:
        result=outputs[method]
        folder=output_path/method
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/"algorithm_result.npz",transformation=result["matrix"],
            reference_keep_mask=result["reference_mask"],moving_keep_mask=result["moving_mask"])
        metadata={key:result[key] for key in ("elapsed_seconds","fit_iterations","selection_info")}
        (folder/"algorithm_info.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
        with (folder/"iterations.csv").open("w",newline="",encoding="utf-8") as file:
            fields=["iteration","nn_rmse_m","used_pairs","effective_weight_sum","stage"]
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(result["history"])
        print(method,"retained",result["reference_mask"].sum(),result["moving_mask"].sum(),"iterations",result["fit_iterations"],flush=True)
    np.savez_compressed(output_path/"input_only_evidence.npz",forward_normal_residual=evidence["residual_m"],
        reverse_normal_residual=evidence["residual_r"],coarse_normal_residual_threshold_m=evidence["threshold"],
        reference_patch_keys=evidence["keys_r"],moving_patch_keys=evidence["keys_m"])
    with (output_path/"patch_statistics.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=list(evidence["patch_statistics"][0]));writer.writeheader();writer.writerows(evidence["patch_statistics"])
    (output_path/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
    audit_record={"dataset_file_accesses":access,"only_coordinate_input_read":bool(access) and all(row["allowed"] for row in access),
                  "input_sha256":hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  "policy":"Audited dataset access after imports/config; all dataset files other than the single coordinate-only inputs.npz are denied"}
    (output_path/"input_access_audit.json").write_text(json.dumps(audit_record,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
