"""Validate saved artifacts and independently cross-check baseline with Open3D."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "numpy_baseline")
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
reference, moving = [np.loadtxt(row["path"])[:, :3] for row in manifest["inputs"]]
checks = {"input_hashes_unchanged": all(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"] for row in manifest["inputs"] + manifest.get("normal_inputs", []))}
for method in metrics:
    if method == "before":
        continue
    with np.load(folder / f"{method}_transformation.npz") as saved:
        transform, origin = saved["affine_transformation"], saved["reduction_point"]
    aligned = np.loadtxt(folder / f"{method}_aligned.xyz")
    reproduced = (moving - origin) @ transform[:3, :3].T + transform[:3, 3] + origin
    # These are tiny samples: use an independent full pairwise evaluation here.
    distances = np.linalg.norm(aligned[:, None, :] - reference[None, :, :], axis=2).min(axis=1)
    checks[f"{method}_transform_roundtrip"] = bool(np.allclose(reproduced, aligned, rtol=0, atol=1e-10))
    checks[f"{method}_rmse_reproduced"] = bool(np.isclose(np.sqrt(np.mean(distances**2)) * 1000, metrics[method]["moving_to_reference_rmse_mm"], rtol=0, atol=1e-7))

if folder.name == "numpy_baseline":
    import open3d as o3d
    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(moving)
    target.points = o3d.utility.Vector3dVector(reference)
    result = o3d.pipelines.registration.registration_icp(
        source, target, 1.0, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-10, relative_rmse=1e-10, max_iteration=100))
    recorded = json.loads((folder / "ordinary_transformation.json").read_text(encoding="utf-8"))
    checks["open3d_transformation_agrees"] = bool(np.allclose(result.transformation, recorded["affine_transformation"], rtol=0, atol=1e-9))
    checks["open3d_rmse_agrees"] = bool(np.isclose(result.inlier_rmse * 1000, metrics["ordinary"]["moving_to_reference_rmse_mm"], atol=1e-7, rtol=0))
    print(f"Open3D {o3d.__version__}: RMSE {result.inlier_rmse * 1000:.9f} mm, fitness {result.fitness}")
report = {"passed": all(checks.values()), "checks": checks, "python": sys.executable}
(folder / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(0 if report["passed"] else 1)
