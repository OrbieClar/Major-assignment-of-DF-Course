"""Reproducible ICP comparison; input files are never modified.

The numpy engine is an explicitly labelled ordinary-ICP baseline, not py4dgeo.
The py4dgeo engine calls its public ordinary and stable-area ICP APIs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("registration")


def timed_python_call(function, timeout, **kwargs):
    """Bound py4dgeo's Python iteration loops without changing its equations."""
    deadline = time.perf_counter() + timeout
    previous_trace = sys.gettrace()
    def trace(frame, event, arg):
        if time.perf_counter() > deadline:
            raise TimeoutError(f"Registration exceeded {timeout:g} seconds")
        return trace
    try:
        sys.settrace(trace)
        return function(**kwargs)
    finally:
        sys.settrace(previous_trace)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def nearest(source, target):
    # Chunked brute force is deliberately limited to the 441-point plane sample.
    # No threshold or residual trimming: all source points enter the evaluation.
    distances, indices = [], []
    for start in range(0, len(source), 256):
        delta = source[start:start + 256, None, :] - target[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", delta, delta)
        idx = d2.argmin(axis=1)
        distances.append(np.sqrt(d2[np.arange(len(idx)), idx]))
        indices.append(idx)
    return np.concatenate(distances), np.concatenate(indices)


def evaluate(moving, reference):
    forward, idx = nearest(moving, reference)
    backward, _ = nearest(reference, moving)
    return {
        "moving_to_reference_rmse_mm": float(np.sqrt(np.mean(forward**2)) * 1000),
        "reference_to_moving_rmse_mm": float(np.sqrt(np.mean(backward**2)) * 1000),
        "symmetric_rmse_mm": float(np.sqrt((np.mean(forward**2) + np.mean(backward**2)) / 2) * 1000),
        "median_mm": float(np.median(forward) * 1000),
        "p95_mm": float(np.quantile(forward, 0.95) * 1000),
        "maximum_mm": float(forward.max() * 1000),
        "matched_points": int(len(forward)),
        "fitness": 1.0,
    }, forward, idx


def apply_transform(points, matrix, reduction):
    return (points - reduction) @ matrix[:3, :3].T + matrix[:3, 3] + reduction


def numpy_icp(reference, moving, max_iterations, tolerance, out):
    cloud = moving.copy()
    total = np.eye(4)
    previous = None
    history = []
    for step in range(max_iterations):
        _, indices = nearest(cloud, reference)
        pairs = reference[indices]
        a, b = cloud.mean(axis=0), pairs.mean(axis=0)
        u, _, vt = np.linalg.svd((cloud - a).T @ (pairs - b))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = b - rotation @ a
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = rotation, translation
        cloud = cloud @ rotation.T + translation
        total = delta @ total
        residual, _ = nearest(cloud, reference)
        rmse = float(np.sqrt(np.mean(residual**2)))
        history.append({"iteration": step + 1, "rmse_m": rmse})
        LOG.info("NumPy ordinary ICP iteration %d: RMSE %.9f mm", step + 1, rmse * 1000)
        if previous is not None and abs(previous - rmse) < tolerance:
            break
        previous = rmse
    with (out / "iterations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iteration", "rmse_m"])
        writer.writeheader()
        writer.writerows(history)
    return total, np.zeros(3), cloud


def save_svg(path, rows):
    names = list(rows)
    max_value = max(rows.values()) or 1
    fragments = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300" viewBox="0 0 900 300">',
                 '<rect width="900" height="300" fill="white"/>',
                 '<text x="25" y="35" font-family="sans-serif" font-size="22">All-point nearest-neighbor RMSE (mm)</text>']
    for i, name in enumerate(names):
        y = 65 + i * 65
        width = rows[name] / max_value * 520
        fragments.extend([f'<text x="25" y="{y+24}" font-family="sans-serif" font-size="17">{name}</text>',
                          f'<rect x="230" y="{y}" width="{width:.2f}" height="35" fill="{["#8293a8", "#3983ba", "#25a18e"][i]}"/>',
                          f'<text x="{240+width:.2f}" y="{y+24}" font-family="sans-serif" font-size="17">{rows[name]:.4f}</text>'])
    fragments.append('</svg>')
    path.write_text("\n".join(fragments), encoding="utf-8")


def run(args, out):
    import numpy
    global np
    np = numpy
    if args.engine == "numpy" and args.method != "ordinary":
        raise ValueError("The numpy engine only implements ordinary ICP; stable-area ICP requires py4dgeo.")
    inputs = [ROOT / "usage_data" / f"plane_horizontal_t{i}.xyz" for i in (1, 2)]
    reference, moving = [np.loadtxt(path, usecols=(0, 1, 2)) for path in inputs]
    for cloud in (reference, moving):
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError("Invalid input coordinates")
        if len(cloud) > 10000:
            raise ValueError("This small-sample evaluation is not intended for the large AHK files")
    packages = {}
    for name in ("numpy", "py4dgeo", "scipy", "matplotlib", "laspy", "lazrs"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "isolated_venv": sys.prefix != sys.base_prefix, "packages": packages,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "parameters": vars(args), "input_unit": "m (assumed, consistent with previous 100.76 mm baseline)",
        "inputs": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "points": len(cloud), "bounds_min": cloud.min(0).tolist(),
                    "bounds_max": cloud.max(0).tolist()} for path, cloud in zip(inputs, (reference, moving))],
        "evaluation": "Unthresholded all-point Euclidean nearest-neighbor distances; fitness=1 by definition, no ground truth.",
    }
    if args.engine == "py4dgeo" and args.normal_source == "provided":
        manifest["normal_inputs"] = [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [ROOT / "usage_data" / f"plane_horizontal_t{i}_w_normals.laz" for i in (1, 2)]]
    write_json(out / "manifest.json", manifest)
    before, _, _ = evaluate(moving, reference)
    LOG.info("Before: %.9f mm (%d / %d points)", before["moving_to_reference_rmse_mm"], len(moving), len(reference))
    outcomes = {"before": before}
    statuses = {}
    pg = None
    if args.engine == "py4dgeo":
        import py4dgeo as pg
        pg.set_py4dgeo_logfile(str(ROOT / "logs" / f"py4dgeo_{out.name}.log"))
        logging.getLogger("py4dgeo").propagate = False
    methods = ["ordinary", "stable"] if args.method == "both" else [args.method]
    for name in methods:
        started = time.perf_counter()
        try:
            if pg is None:
                matrix, reduction, aligned = numpy_icp(reference, moving, args.max_iterations, args.tolerance, out)
            else:
                # Fresh independent epochs ensure neither method starts from the other's result.
                ref_epoch, epoch = pg.Epoch(reference.copy()), pg.Epoch(moving.copy())
                if name == "ordinary":
                    trafo = pg.iterative_closest_point(ref_epoch, epoch, max_iterations=args.max_iterations,
                                                       tolerance=args.tolerance, reduction_point=np.zeros(3))
                else:
                    if args.normal_source == "provided":
                        for i, current in ((1, ref_epoch), (2, epoch)):
                            filename = ROOT / "usage_data" / f"plane_horizontal_t{i}_w_normals.laz"
                            with_normals = pg.read_from_las(str(filename), normal_columns=["NormalX", "NormalY", "NormalZ"])
                            if not np.allclose(current.cloud, with_normals.cloud, rtol=0, atol=1e-8):
                                raise ValueError("Provided normals do not match XYZ point order/coordinates")
                            current.normals_attachment(np.asarray(with_normals.normals).copy())
                    else:
                        ref_epoch.calculate_normals(radius=args.normal_radius)
                        epoch.calculate_normals(radius=args.normal_radius)
                    if not np.isfinite(ref_epoch.normals).all() or not np.isfinite(epoch.normals).all():
                        raise ValueError("Nonfinite normals: revise the neighborhood radius")
                    trafo = timed_python_call(pg.icp_with_stable_areas, args.timeout_seconds,
                        reference_epoch=ref_epoch, epoch=epoch, initial_distance_threshold=args.initial_threshold,
                        level_of_detection=args.lod,
                        reference_supervoxel_resolution=args.supervoxel,
                        supervoxel_resolution=args.supervoxel,
                        min_svp_num=args.min_supervoxel_points, reduction_point=np.zeros(3))
                matrix = np.asarray(trafo.affine_transformation).copy()
                if matrix.shape == (3, 4):
                    matrix = np.vstack((matrix, [0, 0, 0, 1]))
                reduction = np.asarray(trafo.reduction_point).copy()
                epoch.transform(trafo)
                aligned = np.asarray(epoch.cloud).copy()
            if not np.isfinite(matrix).all() or not np.isfinite(aligned).all():
                raise ValueError("Nonfinite registration result")
            if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6):
                raise ValueError("Transformation is not rigid")
            if not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6):
                raise ValueError("Rotation is not proper")
            if not np.allclose(apply_transform(moving, matrix, reduction), aligned, atol=1e-9):
                raise ValueError("Saved transform does not reproduce aligned coordinates")
            after, distances, indices = evaluate(aligned, reference)
            after["elapsed_seconds"] = time.perf_counter() - started
            after["rmse_reduced"] = after["moving_to_reference_rmse_mm"] < before["moving_to_reference_rmse_mm"]
            np.savetxt(out / f"{name}_aligned.xyz", aligned, fmt="%.12f")
            np.savetxt(out / f"{name}_transformation.csv", matrix, delimiter=",", fmt="%.16g")
            np.savez(out / f"{name}_transformation.npz", affine_transformation=matrix, reduction_point=reduction)
            write_json(out / f"{name}_transformation.json", {
                "direction": "epoch_t2_to_reference_t1", "affine_transformation": matrix.tolist(),
                "reduction_point": reduction.tolist(),
                "formula": "aligned = (moving - reduction_point) @ R.T + t + reduction_point"})
            residuals = np.column_stack((np.arange(len(moving)), indices, aligned, distances * 1000))
            np.savetxt(out / f"{name}_residuals.csv", residuals, delimiter=",", fmt="%.12g",
                       header="moving_index,reference_index,x_m,y_m,z_m,distance_mm", comments="")
            # Read exported artifacts back, not just in-memory arrays.
            if not np.allclose(np.loadtxt(out / f"{name}_aligned.xyz"), aligned, atol=1e-10):
                raise ValueError("Export round-trip check failed")
            outcomes[name] = after
            statuses[name] = {"status": "completed"}
            LOG.info("%s: %.9f mm, %.3f s", name, after["moving_to_reference_rmse_mm"], after["elapsed_seconds"])
            if not after["rmse_reduced"]:
                LOG.warning("%s did not reduce the global nearest-neighbor RMSE; inspect geometry and parameters", name)
        except Exception as error:
            LOG.exception("%s failed", name)
            statuses[name] = {"status": "failed", "error": repr(error)}
    write_json(out / "metrics.json", outcomes)
    write_json(out / "methods_status.json", statuses)
    fields = ["method", "moving_to_reference_rmse_mm", "reference_to_moving_rmse_mm", "symmetric_rmse_mm", "median_mm", "p95_mm", "maximum_mm", "matched_points", "fitness", "elapsed_seconds", "rmse_reduced"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"method": key, **value} for key, value in outcomes.items())
    save_svg(out / "rmse_comparison.svg", {name: row["moving_to_reference_rmse_mm"] for name, row in outcomes.items()})
    if any(row["status"] != "completed" for row in statuses.values()):
        raise RuntimeError("One or more methods failed; see methods_status.json and log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["numpy", "py4dgeo"], default="py4dgeo")
    parser.add_argument("--method", choices=["ordinary", "stable", "both"], default="both")
    parser.add_argument("--output", default="py4dgeo_comparison")
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--normal-radius", type=float, default=2.5)
    parser.add_argument("--normal-source", choices=["estimate", "provided"], default="provided")
    parser.add_argument("--initial-threshold", type=float, default=0.2)
    parser.add_argument("--lod", type=float, default=0.015)
    parser.add_argument("--supervoxel", type=float, default=4.0)
    parser.add_argument("--min-supervoxel-points", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=15)
    args = parser.parse_args()
    if Path(args.output).name != args.output:
        parser.error("--output must be a single directory name")
    if args.initial_threshold < args.lod or args.lod <= 0:
        parser.error("Require initial-threshold >= lod > 0")
    out = ROOT / "results" / args.output
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / "logs" / f"{args.output}.log", encoding="utf-8")])
    try:
        run(args, out)
    except Exception as error:
        LOG.exception("Run failed")
        write_json(out / "run_status.json", {"status": "failed", "error": repr(error),
                   "traceback": traceback.format_exc(), "python": sys.executable, "arguments": vars(args)})
        return 1
    write_json(out / "run_status.json", {"status": "completed", "python": sys.executable, "arguments": vars(args)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
