"""Build paired, reproducible terrain registration datasets with separate truth.

Uses only the saved AHK terrain as a geometric template. New epoch differences
are synthetic; these are not the measured changes between 2017 and 2018.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("terrain_benchmark")
BLUE, ORANGE, GREEN, RED = (40, 135, 235), (245, 130, 35), (50, 175, 100), (225, 65, 55)


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_ascii_ply(path):
    properties = []
    with path.open(encoding="ascii") as file:
        if file.readline().strip() != "ply" or file.readline().strip() != "format ascii 1.0":
            raise ValueError("Expected an ASCII PLY")
        n = None
        for line in file:
            if line.startswith("element vertex "):
                n = int(line.split()[-1])
            elif line.startswith("property "):
                properties.append(line.split()[-1])
            elif line.strip() == "end_header":
                break
        values = np.loadtxt(file)
    xyz = values[:, [properties.index(name) for name in ("x", "y", "z")]]
    if len(xyz) != n or not np.isfinite(xyz).all():
        raise ValueError("Invalid PLY point count or coordinates")
    return xyz, properties


def write_ply(path, points, colors, fields=None):
    points = np.asarray(points)
    colors = np.broadcast_to(np.asarray(colors, dtype=np.uint8), (len(points), 3))
    fields = fields or {}
    header = ["ply", "format ascii 1.0", "comment Local coordinates in meters; see metadata.json for original origin",
              f"element vertex {len(points)}", "property double x", "property double y", "property double z",
              "property uchar red", "property uchar green", "property uchar blue"]
    header.extend(f"property double scalar_{name}" for name in fields)
    header.append("end_header")
    data = np.column_stack((points, colors, *fields.values()))
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("\n".join(header) + "\n")
        np.savetxt(file, data, fmt=["%.12f"] * 3 + ["%d"] * 3 + ["%.12f"] * len(fields))


def overlay(path, reference, moving):
    write_ply(path, np.vstack((reference, moving)),
              np.vstack((np.tile(BLUE, (len(reference), 1)), np.tile(ORANGE, (len(moving), 1)))),
              {"epoch_id": np.r_[np.ones(len(reference)), np.full(len(moving), 2)]})


def apply(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def bump(xy, center, radius, height):
    r2 = np.sum((xy - center)**2, axis=1) / radius**2
    return height * np.maximum(1.0 - r2, 0.0)**2


def choose_center(master, radius):
    xy = master[:, :2]
    hull = ConvexHull(xy)
    tree = cKDTree(xy)
    midpoint = (xy.min(0) + xy.max(0)) / 2
    candidates = np.array(np.meshgrid(*[np.linspace(lo, hi, 13) for lo, hi in zip(
        np.quantile(xy, .2, axis=0), np.quantile(xy, .8, axis=0))])).reshape(2, -1).T
    angles = np.linspace(0, 2*np.pi, 80, endpoint=False)
    probes = np.vstack([np.column_stack((np.cos(angles), np.sin(angles))) * r
                        for r in np.linspace(0, radius, 12)])
    valid = []
    for candidate in candidates:
        clearance = np.min(-(hull.equations[:, :2] @ candidate + hull.equations[:, 2]))
        if clearance < radius + .5:
            continue
        spacing, _ = tree.query(candidate + probes)
        if spacing.max() > .65:
            continue
        valid.append((np.linalg.norm(candidate-midpoint), candidate, float(clearance), float(spacing.max())))
    if not valid:
        raise ValueError("No fully supported disk found; reduce radius or inspect the chosen terrain")
    _, center, clearance, max_gap = min(valid, key=lambda row: row[0])
    return center, {"convex_hull_clearance_m": clearance, "maximum_probe_distance_to_master_xy_m": max_gap}


def ordinary_icp(reference, moving, max_iterations=150, tolerance=1e-9):
    """Same proper Kabsch fit as verified baseline, accelerated by exact KDTree."""
    tree = cKDTree(reference)
    cloud, total = moving.copy(), np.eye(4)
    previous = None
    history = []
    for step in range(max_iterations):
        _, index = tree.query(cloud, k=1)
        target = reference[index]
        c0, c1 = cloud.mean(0), target.mean(0)
        u, _, vt = np.linalg.svd((cloud-c0).T @ (target-c1))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        delta = np.eye(4)
        delta[:3, :3], delta[:3, 3] = rotation, c1 - rotation @ c0
        cloud = apply(cloud, delta)
        total = delta @ total
        distance, _ = tree.query(cloud)
        rmse = float(np.sqrt(np.mean(distance**2)))
        history.append({"iteration": step+1, "all_point_nn_rmse_m": rmse})
        if previous is not None and abs(previous-rmse) < tolerance:
            break
        previous = rmse
    return total, cloud, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="ahk2017_spatial_020m.ply")
    parser.add_argument("--output", default="terrain_benchmark_v1")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--radius", type=float, default=4.5)
    parser.add_argument("--height", type=float, default=.5)
    args = parser.parse_args()
    if Path(args.output).name != args.output or args.radius <= 0 or args.height <= 0:
        parser.error("Use a simple output name and positive bump radius/height")
    out = ROOT / "results" / args.output
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT/"logs"/f"{args.output}.log", encoding="utf-8")])
    source = ROOT / args.source
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    global_master, properties = read_ascii_ply(source)
    origin = global_master.mean(0)
    master = global_master - origin
    if len(np.unique(master[:, :2], axis=0)) != len(master):
        raise ValueError("Duplicate XY locations need explicit height-field aggregation before volume modeling")
    center, support = choose_center(master, args.radius)
    LOG.info("Source points %d; local origin %s; bump center XY %s",len(master), origin, center)
    # A deterministic piecewise-linear height field defines the synthetic baseline surface.
    triangulation = Delaunay(master[:, :2])
    surface = LinearNDInterpolator(triangulation, master[:, 2])
    noise_sigma = .005
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(args.seed).spawn(5)]
    sampling, ref_noise_rng, mov_noise_rng, check_rng, order_rng = streams
    holdout_mask = sampling.random(len(master)) < .10
    available = np.flatnonzero(~holdout_mask)
    ref_ids = available[sampling.random(len(available)) < .90]
    mov_ids = available[sampling.random(len(available)) < .90]
    ref_ids = order_rng.permutation(ref_ids)
    mov_ids = order_rng.permutation(mov_ids)
    # Same indices and measurement noise in both cases isolate the effect of deformation.
    reference_noise = ref_noise_rng.normal(0, noise_sigma, (len(ref_ids), 3))
    moving_noise = mov_noise_rng.normal(0, noise_sigma, (len(mov_ids), 3))
    reference = master[ref_ids] + reference_noise
    radius_to_center = np.linalg.norm(master[:, :2]-center, axis=1)
    check_ids = np.flatnonzero(holdout_mask & (radius_to_center > args.radius + .75))
    check_target = master[check_ids]
    check_noise = check_rng.normal(0, noise_sigma, check_target.shape)
    forward = np.eye(4)
    angles = [.4, -.3, 1.0]
    forward[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    forward[:3, 3] = [.3, -.2, .15]
    inverse = np.linalg.inv(forward)
    inverse_global = inverse.copy()
    inverse_global[:3, 3] = origin + inverse[:3, 3] - inverse[:3, :3] @ origin
    clean_check_measured = apply(check_target, forward)
    noisy_check_measured = apply(check_target+check_noise, forward)
    assert not np.intersect1d(check_ids, np.r_[ref_ids, mov_ids]).size
    write_ply(out/"master_local_height.ply", master,
              (plt.get_cmap("terrain")((master[:,2]-master[:,2].min()) / np.ptp(master[:,2]))[:,:3]*255).astype(np.uint8),
              {"height_local_m":master[:,2]})
    metadata = {
        "source":str(source), "source_sha256":source_hash, "master_points":len(master),
        "source_properties":properties, "source_interpretation":"Use exported coordinates as the mother geometry; no assumption they are an unchanged subset of the original LAZ",
        "coordinate_unit":"m; inherited from AHK metric coordinates", "global_origin":origin.tolist(),
        "local_coordinate_formula":"local = exported_source_xyz - global_origin; source_xyz = local + global_origin",
        "original_crs":"Not encoded in the source PLY; no new CRS is assigned",
        "seed":args.seed, "sampling":"10% master vertices withheld; reference and moving independently retain 90% of remaining vertices; shuffled point order",
        "noise_sigma_per_axis_m":noise_sigma, "noise_location":"Independent isotropic XYZ Gaussian noise in terrain frame before known rigid transform",
        "reference_points":len(reference), "moving_points":len(mov_ids), "stable_holdout_points":len(check_ids),
        "holdout_boundary_buffer_m":.75, "shared_source_vertices_in_inputs":len(np.intersect1d(ref_ids,mov_ids)),
        "forward_euler_xyz_degrees":angles, "forward_translation_local_m":forward[:3,3].tolist(),
        "forward_transform_local":forward.tolist(), "registration_truth_local":inverse.tolist(),
        "registration_truth_global":inverse_global.tolist(), "transformation_convention":"column-vector homogeneous; row arrays use xyz @ R.T + t",
        "bump":{"center_local_xy_m":center.tolist(),"center_source_xy_m":(center+origin[:2]).tolist(),
                "radius_m":args.radius,"peak_height_m":args.height,
                "formula":"delta_z = h * max(1 - ((x-cx)^2+(y-cy)^2)/r^2, 0)^2",
                "volume_formula":"V = pi * r^2 * h / 3", **support},
        "surface_model":"Piecewise-linear Delaunay interpolation of master XY/Z over its convex hull; synthetic geometry, not an assertion of measured mass change",
        "dropped_source_attributes":"Original normals, intensity and high-level scan fields are not transferred to synthetic inputs; recompute normals if needed",
        "packages":{name:importlib.metadata.version(name) for name in ("numpy","scipy","matplotlib")},
        "python":sys.executable,
    }
    dump(out/"metadata.json",metadata)
    all_metrics = []
    validation = {"input_is_finite":bool(np.isfinite(master).all()), "holdout_disjoint":True,
                  "rigid_rotation_proper":bool(np.allclose(forward[:3,:3].T@forward[:3,:3],np.eye(3),atol=1e-12)),
                  "truth_transform_is_inverse":bool(np.allclose(inverse@forward,np.eye(4),atol=1e-12))}
    for case, height in (("01_rigid_only",0.0),("02_local_deposition",args.height)):
        folder = out/case
        truth_folder = folder/"truth"
        truth_folder.mkdir(parents=True,exist_ok=True)
        dz = bump(master[mov_ids,:2],center,args.radius,height)
        changed = dz > 0
        moving_noiseless_aligned = master[mov_ids].copy()
        moving_noiseless_aligned[:,2] += dz
        truth_aligned = moving_noiseless_aligned + moving_noise
        moving = apply(truth_aligned,forward)
        # Algorithm inputs deliberately contain no true correspondences, masks or transformations.
        np.savez_compressed(folder/"inputs.npz",reference=reference,moving=moving)
        np.savetxt(folder/"reference.xyz",reference,fmt="%.12f")
        np.savetxt(folder/"moving.xyz",moving,fmt="%.12f")
        write_ply(folder/"reference_blue.ply",reference,BLUE)
        write_ply(folder/"moving_orange.ply",moving,ORANGE)
        overlay(folder/"01_before_overlay.ply",reference,moving)
        overlay(truth_folder/"02_after_true_transform_overlay.ply",reference,truth_aligned)
        write_ply(truth_folder/"moving_regions_after_true_transform.ply",truth_aligned,
                  np.where(changed[:,None],np.asarray(RED),np.asarray(GREEN)),
                  {"true_change_m":dz,"changed":changed.astype(float)})
        write_ply(truth_folder/"stable_holdout_checkpoints.ply",check_target,(160,70,205))
        np.savez_compressed(truth_folder/"ground_truth.npz",reference_source_indices=ref_ids,moving_source_indices=mov_ids,
            moving_changed_mask=changed,moving_true_dz=dz,moving_true_aligned_noisy=truth_aligned,
            moving_true_aligned_noiseless=moving_noiseless_aligned,reference_noiseless=master[ref_ids],
            reference_noise=reference_noise,moving_noise=moving_noise,stable_check_source_indices=check_ids,
            stable_check_target=check_target,stable_check_moving_noiseless=clean_check_measured,
            stable_check_moving_noisy=noisy_check_measured,forward_transform=forward,correct_registration=inverse,
            global_origin=origin,reference_stable_mask=np.linalg.norm(master[ref_ids,:2]-center,axis=1)>args.radius+.75)
        np.savetxt(truth_folder/"correct_registration_local.csv",inverse,delimiter=",",fmt="%.16g")
        np.savetxt(truth_folder/"correct_registration_global.csv",inverse_global,delimiter=",",fmt="%.16g")
        # Integrate on a fixed rectangular grid containing the entire support disk.
        # Cell-centered quadrature is an independent check, not the analytic definition.
        cell = .1
        count = int(np.ceil((2*args.radius+2)/cell))
        axes = [center[i]-(count*cell)/2+(np.arange(count)+.5)*cell for i in (0,1)]
        gx,gy = np.meshgrid(*axes,indexing="xy")
        grid_xy = np.column_stack((gx.ravel(),gy.ravel()))
        grid_base = np.asarray(surface(grid_xy))
        grid_dz = bump(grid_xy,center,args.radius,height)
        assert np.isfinite(grid_base[grid_dz>0]).all()
        volume = float(np.pi*args.radius**2*height/3)
        quadrature = float(grid_dz.sum()*cell**2)
        np.savez_compressed(truth_folder/"volume_grid.npz",xy_local=grid_xy,z_reference=grid_base,
            z_second=grid_base+grid_dz,true_dz=grid_dz,valid_surface=np.isfinite(grid_base),
            cell_size_m=cell,grid_shape=gx.shape,area_per_cell_m2=cell**2)
        case_truth = {"case":case,"synthetic":True,"true_volume_m3":volume,"numerical_volume_m3":quadrature,
                      "quadrature_absolute_error_m3":abs(volume-quadrature),"changed_moving_points":int(changed.sum()),
                      "changed_moving_fraction":float(changed.mean()),"true_peak_height_m":height,
                      "volume_definition":"Vertical volume between the clean height fields over the complete circular support; not a point-count estimate",
                      "input_fields_allowed_for_methods":["reference","moving"],
                      "truth_access":"Masks, source indices, correct poses and check points are only for evaluation/oracle controls, never for proposed-method fitting"}
        dump(truth_folder/"truth.json",case_truth)
        started=time.perf_counter()
        matrix,aligned,history=ordinary_icp(reference,moving)
        elapsed=time.perf_counter()-started
        tree=cKDTree(reference)
        before_dist=tree.query(moving)[0]
        after_dist=tree.query(aligned)[0]
        pose_rotation=matrix[:3,:3]@inverse[:3,:3].T
        angle_error=float(np.degrees(np.arccos(np.clip((np.trace(pose_rotation)-1)/2,-1,1))))
        pose_distance=float(np.linalg.norm(matrix[:3,3]-inverse[:3,3]))
        check_residual=apply(clean_check_measured,matrix)-check_target
        noisy_check_residual=apply(noisy_check_measured,matrix)-check_target
        truth_residual=aligned-truth_aligned
        metric={"case":case,"method":"ordinary_point_to_point_icp_numpy_kabsch_scipy_exact_kdtree",
                "reference_points":len(reference),"moving_points":len(moving),"iterations":len(history),
                "rotation_error_deg":angle_error,"translation_error_mm":pose_distance*1000,
                "stable_holdout_pose_rmse_mm":float(np.sqrt(np.mean(np.sum(check_residual**2,axis=1)))*1000),
                "stable_holdout_noisy_rmse_mm":float(np.sqrt(np.mean(np.sum(noisy_check_residual**2,axis=1)))*1000),
                "all_point_nn_before_rmse_mm":float(np.sqrt(np.mean(before_dist**2))*1000),
                "all_point_nn_after_rmse_mm":float(np.sqrt(np.mean(after_dist**2))*1000),
                "true_volume_m3":volume,"changed_point_count":int(changed.sum()),"elapsed_seconds":elapsed,
                "change_region_pose_height_mae_mm":float(np.mean(abs(truth_residual[changed,2]))*1000) if changed.any() else None}
        dump(folder/"ordinary_icp_metrics.json",metric)
        np.savetxt(folder/"ordinary_icp_transform.csv",matrix,delimiter=",",fmt="%.16g")
        np.savetxt(folder/"ordinary_icp_aligned.xyz",aligned,fmt="%.12f")
        write_ply(folder/"ordinary_icp_aligned_orange.ply",aligned,ORANGE)
        overlay(folder/"03_after_ordinary_icp_overlay.ply",reference,aligned)
        with (folder/"ordinary_icp_iterations.csv").open("w",encoding="utf-8",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
        # Reload public inputs/outputs and truth to verify the delivered artifacts.
        with np.load(folder/"inputs.npz") as saved:
            restored=apply(saved["moving"],inverse)
            validation[f"{case}_inverse_recovers_noisy_deformed_epoch"]=bool(np.allclose(restored,truth_aligned,atol=1e-10,rtol=0))
        reloaded_ply,_=read_ascii_ply(folder/"03_after_ordinary_icp_overlay.ply")
        validation[f"{case}_ply_roundtrip"]=bool(np.allclose(reloaded_ply,np.vstack((reference,aligned)),atol=1e-10,rtol=0))
        validation[f"{case}_volume_quadrature"]=abs(volume-quadrature)<.002
        validation[f"{case}_transform_reproduces_export"]=bool(np.allclose(apply(moving,matrix),np.loadtxt(folder/"ordinary_icp_aligned.xyz"),atol=1e-10,rtol=0))
        all_metrics.append(metric)
        LOG.info("%s: %d changed points; true volume %.8f m3; rotation error %.6f deg; translation error %.3f mm; stable pose RMSE %.3f mm",
                 case,int(changed.sum()),volume,angle_error,pose_distance*1000,metric["stable_holdout_pose_rmse_mm"])
    validation["source_hash_unchanged"]=hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    validation["rigid_only_baseline_rotation_below_005deg"]=all_metrics[0]["rotation_error_deg"]<.05
    validation["rigid_only_baseline_translation_below_10mm"]=all_metrics[0]["translation_error_mm"]<10
    dump(out/"validation.json",{"passed":all(validation.values()),"checks":validation})
    dump(out/"baseline_summary.json",all_metrics)
    with (out/"baseline_summary.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(all_metrics[0]));writer.writeheader();writer.writerows(all_metrics)
    # Plot geometry and true changes, not a generative illustration.
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    pts=axes[0].scatter(master[:,0],master[:,1],c=master[:,2]+origin[2],s=2,cmap="terrain",rasterized=True)
    fig.colorbar(pts,ax=axes[0],label="Source Z (m)")
    axes[0].add_patch(plt.Circle(center,args.radius,fill=False,color="red",linewidth=1.5))
    axes[0].set_title("Saved terrain and prescribed deposition footprint")
    pts=axes[1].scatter(master[:,0],master[:,1],c=bump(master[:,:2],center,args.radius,args.height),s=2,cmap="YlOrRd",vmin=0,vmax=args.height,rasterized=True)
    fig.colorbar(pts,ax=axes[1],label="True vertical change (m)")
    axes[1].set_title("Clean synthetic change field")
    for ax in axes:ax.set_aspect("equal");ax.set_xlabel("Local X (m)");ax.set_ylabel("Local Y (m)")
    fig.savefig(out/"terrain_and_change.png",dpi=170);plt.close(fig)
    lines=["# 有真值的 AHK 地形实验数据", "", "已生成两组半合成数据：真实地形形状来自用户导出的 2017 样区，时相差异由程序构造。它们不是实测 2017/2018 两期变化。", "",
        f"母本 {len(master):,} 点；参考时相 {len(reference):,} 点；待配准时相 {len(mov_ids):,} 点；独立稳定检查点 {len(check_ids):,} 点。两期独立抽样并打乱点序，分别加入每轴标准差 5 mm 的高斯噪声。两组共用抽样、噪声和刚体偏移。", "",
        "## 已知变换与变化", "", "所有输出使用同一个局部坐标系，单位米；原点及恢复导出源文件坐标的公式见 metadata.json。原 PLY 没有 CRS 元信息，没有新指定坐标系，也没有覆盖原文件。", "",
        "先在地形坐标中加入局部变化和噪声，再施加绕局部原点的 XYZ 欧拉角 [0.4, -0.3, 1.0] 度（SciPy 小写 xyz 约定）和位移 [0.30, -0.20, 0.15] 米。正确配准为该变换的逆，绝不是简单把角度和位移同时取负。矩阵是使用真值的权威方式。", "",
        f"局部堆积是半径 {args.radius:g} m、最大高度 {args.height:g} m 的圆形平滑隆起：`dz = h * max(1-rho²/r²,0)²`。母本的 XY/Z 三角网定义基准高程曲面；改变的是竖直高程，忽略悬挑和多值高程。解析体积 `V = pi*r²*h/3 = {np.pi*args.radius**2*args.height/3:.9f} m³`，并以 0.1 m 网格积分交叉检查。它是定义的无噪声合成体积，不是自然地形真实变化量，也不是按变化点数估算的体积。", "",
        "## 文件", "", "| 路径 | 内容 |", "| --- | --- |", "| metadata.json | 来源哈希、坐标原点、完整参数、变换、环境 |",
        "| master_local_height.ply | 地形母本，按高程着色 |", "| 01_rigid_only/ | 纯刚性偏移测试 |", "| 02_local_deposition/ | 局部堆积加刚性偏移测试 |",
        "| 各组 inputs.npz、reference.xyz、moving.xyz | 方法实际输入，仅含坐标 |", "| 各组 reference_blue.ply、moving_orange.ply | 两期分别查看 |",
        "| 各组 01_before_overlay.ply | 配准前，蓝=t1，橙=t2 |", "| 各组 03_after_ordinary_icp_overlay.ply | 普通 ICP 实际输出 |",
        "| 各组 truth/02_after_true_transform_overlay.ply | 正确变换后的预期效果，局部堆积仍保留 |",
        "| 各组 truth/moving_regions_after_true_transform.ply | 已知变化区红色，稳定区绿色 |",
        "| 各组 truth/ground_truth.npz | 变换、点源编号、噪声、变化标签、独立检查点 |", "| 各组 truth/volume_grid.npz、truth.json | 高程模型网格、解析与数值体积真值 |",
        "| baseline_summary.csv/json | 普通 ICP 试跑结果 |", "| validation.json | 文件回读、变换、体积及纯配准精度检查 |", "",
        "## 普通 ICP 试跑", "", "使用已验证的正确 Kabsch 旋转/平移计算，最近邻查询改为 SciPy 精确 KDTree 以支持万级点云。没有调用存在问题的 py4dgeo 1.2.0 配准实现，没有在拟合中使用变化真值或独立检查点。两组从单位初始矩阵出发，使用全部输入点，无残差剔除，最多 150 次迭代。", "",
        "| 测试组 | 旋转误差/度 | 平移误差/mm | 独立稳定区位姿残差/mm | 全点最近邻 RMSE/mm | 真值体积/m³ |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for m in all_metrics:
        lines.append(f"| {m['case']} | {m['rotation_error_deg']:.6f} | {m['translation_error_mm']:.3f} | {m['stable_holdout_pose_rmse_mm']:.3f} | {m['all_point_nn_after_rmse_mm']:.3f} | {m['true_volume_m3']:.6f} |")
    lines += ["", "独立稳定区位姿残差使用未加噪的留出点，刻画变换误差；另存的含噪声检查区 RMS 包含 5 mm/轴噪声。全点最近邻距离受独立抽样和真实堆积影响，不等于位姿误差，也不能以堆积区距离越小越好作为目标。本阶段只生成体积真值，尚未实现从配准点云反算体积的估计器。", "",
        "## 使用规则与复现", "", "算法只读取 inputs.npz 中的 reference/moving，不能把 truth/ 的掩膜、正确变换、点源编号或检查点用于参考区筛选/拟合。真值仅用于评价和明确标注的理想对照。原有法向量不沿用；需要时在相应点云上重新估计。", "",
        "堆积位置按输入几何覆盖确定，未根据算法结果调整；只有这一处堆积和一个随机种子，结果不能替代鲁棒性实验。固定参数后续所有方法共用。", "",
        "在项目根目录执行：", "", "```powershell", f"& .\\.venv\\Scripts\\python.exe scripts\\build_terrain_benchmark.py --seed {args.seed}", "```", "",
        "代码：scripts/build_terrain_benchmark.py；日志：logs/terrain_benchmark_v1.log。",
        "", f"验证状态：{'全部通过' if all(validation.values()) else '有未通过项，见 validation.json'}。"]
    (out/"数据说明.md").write_text("\n".join(lines),encoding="utf-8")
    LOG.info("Output %s; validation %s",out,all(validation.values()))
    if not all(validation.values()):
        raise RuntimeError("Validation failed; inspect validation.json before using the data")


if __name__=="__main__":
    main()
