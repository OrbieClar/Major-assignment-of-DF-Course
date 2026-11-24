"""Controlled diagnosis: all points vs oracle stable region vs matched random subsets.

Oracle labels and counts are evaluation controls, never an automatic method.
All fitting calls receive coordinates only, start at identity, and use the same
proper Kabsch + exact KDTree ordinary ICP implementation and stopping settings.
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
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_terrain_benchmark import ordinary_icp, apply, overlay, write_ply, read_ascii_ply

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("stable_region_diagnosis")
CASES = ("01_rigid_only", "02_local_deposition")
METHODS = ("all_points", "oracle_stable", "random_matched")


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(reference, moving, matrix, truth):
    correct = truth["correct_registration"]
    aligned = apply(moving, matrix)
    angle = float(np.degrees(np.arccos(np.clip((np.trace(matrix[:3, :3] @ correct[:3, :3].T)-1)/2, -1, 1))))
    translation = float(np.linalg.norm(matrix[:3, 3]-correct[:3, 3])*1000)
    check_error = apply(truth["stable_check_moving_noiseless"], matrix)-truth["stable_check_target"]
    noisy_check_error = apply(truth["stable_check_moving_noisy"], matrix)-truth["stable_check_target"]
    nn = cKDTree(reference).query(aligned)[0]
    changed = truth["moving_changed_mask"]
    dz_error = aligned[:, 2]-truth["moving_true_aligned_noisy"][:, 2]
    change_mae = float(np.mean(np.abs(dz_error[changed]))*1000) if changed.any() else None
    metrics = {
        "rotation_error_deg": angle,
        "translation_error_mm": translation,
        "stable_holdout_pose_rmse_mm": float(np.sqrt(np.mean(np.sum(check_error**2, axis=1)))*1000),
        "stable_holdout_noisy_rmse_mm": float(np.sqrt(np.mean(np.sum(noisy_check_error**2, axis=1)))*1000),
        "all_point_nn_after_rmse_mm": float(np.sqrt(np.mean(nn**2))*1000),
        "change_region_pose_height_mae_mm": change_mae,
    }
    return metrics, aligned, np.linalg.norm(check_error, axis=1)*1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="terrain_benchmark_v1")
    parser.add_argument("--output", default="stable_region_diagnosis_v1")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.dataset, args.output)) or args.repeats < 2:
        parser.error("Use simple folder names and at least two random repeats")
    dataset = ROOT / "results" / args.dataset
    out = ROOT / "results" / args.output
    out.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / "logs" / f"{args.output}.log", encoding="utf-8")])
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(metadata["bump"]["center_local_xy_m"])
    exclusion_radius = metadata["bump"]["radius_m"] + metadata["holdout_boundary_buffer_m"]
    common_truth = np.load(dataset / CASES[1] / "truth" / "ground_truth.npz")
    # Fixed region exclusion in both cases preserves the geometry of the control.
    reference_mask = common_truth["reference_stable_mask"]
    moving_mask = np.linalg.norm(common_truth["moving_true_aligned_noiseless"][:, :2]-center, axis=1) > exclusion_radius
    oracle_ref, oracle_mov = np.flatnonzero(reference_mask), np.flatnonzero(moving_mask)
    common_truth.close()
    with np.load(dataset / CASES[0] / "inputs.npz") as saved:
        total_ref, total_mov = len(saved["reference"]), len(saved["moving"])
    seeds = np.random.SeedSequence(args.seed).spawn(args.repeats)
    random_selections = []
    for child in seeds:
        rng = np.random.default_rng(child)
        random_selections.append((np.sort(rng.choice(total_ref, len(oracle_ref), replace=False)),
                                  np.sort(rng.choice(total_mov, len(oracle_mov), replace=False))))
    source_files = [dataset / "metadata.json", ROOT / "scripts" / "build_terrain_benchmark.py"]
    source_files += [dataset / case / name for case in CASES for name in ("inputs.npz", "truth/ground_truth.npz", "ordinary_icp_transform.csv")]
    input_hashes = {str(path): sha256(path) for path in source_files}
    manifest = {
        "dataset": str(dataset), "source_sha256": input_hashes,
        "solver": "build_terrain_benchmark.ordinary_icp: NumPy proper Kabsch + scipy.spatial.cKDTree exact NN",
        "initial_transform": np.eye(4).tolist(), "max_iterations": 150, "tolerance_rmse_m": 1e-9,
        "correspondences": "Unthresholded point-to-point nearest neighbors; no truth correspondences, no weighting",
        "repeats": args.repeats, "random_seed": args.seed,
        "random_seed_sequence": "NumPy SeedSequence(seed).spawn(repeats), same selections for both paired cases",
        "selected_reference_points": len(oracle_ref), "selected_moving_points": len(oracle_mov),
        "oracle_exclusion_center_local_xy_m": center.tolist(), "oracle_exclusion_radius_m": exclusion_radius,
        "oracle_label_access": "Uses true untransformed XY and reference stable mask only to define the ideal control selection",
        "pure_case_control": "Exclude the same potential-change footprint even though all points are physically unchanged in the rigid-only case",
        "random_label_access": "Only matched sample counts are inherited from the oracle; indices chosen uniformly, independently for reference/moving",
        "evaluation_only_truth": "Correct pose, latent IDs, change heights and independent stable check points are never passed to fitting",
        "coordinate_system": "Same local meter coordinates as terrain_benchmark metadata; plots/PLYs use no offsets or exaggeration",
        "global_origin": metadata["global_origin"],
        "timing": "Fitting only; excludes subset selection, truth evaluation, exports and file I/O",
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
        "python": sys.executable,
    }
    dump(out / "manifest.json", manifest)
    validations = {}
    all_rows = []
    matrices = {}
    errors = {}
    for case in CASES:
        LOG.info("Starting %s: oracle retains %d reference / %d moving points", case, len(oracle_ref), len(oracle_mov))
        case_out = out / case
        case_out.mkdir(exist_ok=True)
        with np.load(dataset / case / "inputs.npz") as saved:
            reference, moving = saved["reference"].copy(), saved["moving"].copy()
        with np.load(dataset / case / "truth" / "ground_truth.npz") as saved:
            truth = {key: saved[key].copy() for key in saved.files}
        validations[f"{case}_oracle_excludes_changed_points"] = not bool(truth["moving_changed_mask"][oracle_mov].any())
        validations[f"{case}_holdout_disjoint_from_input_sources"] = not bool(np.intersect1d(
            truth["stable_check_source_indices"], np.r_[truth["reference_source_indices"], truth["moving_source_indices"]]).size)
        write_ply(case_out / "oracle_reference_selection.ply", reference,
                  np.where(reference_mask[:, None], np.asarray([45, 180, 95]), np.asarray([220, 65, 50])),
                  {"selected_for_oracle_fit": reference_mask.astype(float)})
        # Display-only truth alignment allows a transparent inspection of excluded regions.
        write_ply(case_out / "oracle_moving_selection_truth_aligned.ply", truth["moving_true_aligned_noisy"],
                  np.where(moving_mask[:, None], np.asarray([45, 180, 95]), np.asarray([220, 65, 50])),
                  {"selected_for_oracle_fit": moving_mask.astype(float), "true_change_m": truth["moving_true_dz"]})
        controls = [("all_points", -1, np.arange(total_ref), np.arange(total_mov)),
                    ("oracle_stable", -1, oracle_ref, oracle_mov)]
        controls += [("random_matched", repeat, ref_idx, mov_idx)
                     for repeat, (ref_idx, mov_idx) in enumerate(random_selections)]
        for method, repeat, ref_idx, mov_idx in controls:
            run_name = method if repeat < 0 else f"random_matched/seed_{repeat:02d}"
            run_out = case_out / run_name
            run_out.mkdir(parents=True, exist_ok=True)
            # Only selected coordinates enter the shared solver.
            started = time.perf_counter()
            matrix, _, history = ordinary_icp(reference[ref_idx], moving[mov_idx], max_iterations=150, tolerance=1e-9)
            elapsed = time.perf_counter()-started
            metrics, aligned, check_error = evaluate(reference, moving, matrix, truth)
            row = {"case": case, "method": method, "repeat_index": repeat,
                   "fit_reference_points": len(ref_idx), "fit_moving_points": len(mov_idx),
                   "iterations": len(history), **metrics, "fit_elapsed_seconds": elapsed}
            dump(run_out / "metrics.json", row)
            np.savetxt(run_out / "transformation.csv", matrix, delimiter=",", fmt="%.16g")
            np.savez_compressed(run_out / "fit_selection.npz", reference_indices=ref_idx, moving_indices=mov_idx)
            save_csv(run_out / "iterations.csv", history)
            all_rows.append(row)
            matrices[(case, method, repeat)] = matrix
            if method != "random_matched":
                errors[(case, method)] = check_error
            validations[f"{case}_{method}_{repeat}_proper_rigid_transform"] = bool(
                np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), rtol=0, atol=1e-8)
                and np.isclose(np.linalg.det(matrix[:3, :3]), 1, rtol=0, atol=1e-8))
            if method == "random_matched":
                validations[f"{case}_random_{repeat}_matched_counts"] = len(ref_idx)==len(oracle_ref) and len(mov_idx)==len(oracle_mov)
            if method == "all_points":
                original = np.loadtxt(dataset / case / "ordinary_icp_transform.csv", delimiter=",")
                validations[f"{case}_all_points_reproduces_original_baseline"] = bool(np.allclose(matrix, original, rtol=0, atol=1e-10))
            # Export full transformed cloud, not just the points used to estimate pose.
            # Random example is predeclared repeat 0, never the best/worst seed.
            if repeat < 0 or repeat == 0:
                np.savetxt(run_out / "full_moving_aligned.xyz", aligned, fmt="%.12f")
                overlay(run_out / "after_overlay.ply", reference, aligned)
                write_ply(run_out / "moving_aligned.ply", aligned, (245, 130, 35))
                reloaded, _ = read_ascii_ply(run_out / "after_overlay.ply")
                validations[f"{case}_{method}_{repeat}_ply_roundtrip"] = bool(np.allclose(
                    reloaded, np.vstack((reference, aligned)), rtol=0, atol=1e-10))
            # Read actual saved matrix and selection back, independently rerun evaluation.
            stored = np.loadtxt(run_out / "transformation.csv", delimiter=",")
            reevaluated, _, _ = evaluate(reference, moving, stored, truth)
            validations[f"{case}_{method}_{repeat}_saved_metrics_reproduce"] = all(
                metrics[key] is None or np.isclose(metrics[key], reevaluated[key], rtol=0, atol=1e-8)
                for key in metrics)
            if method != "random_matched" or (repeat+1) % 5 == 0:
                LOG.info("%s %s repeat %d: translation %.4f mm; stable pose %.4f mm", case, method, repeat,
                         metrics["translation_error_mm"], metrics["stable_holdout_pose_rmse_mm"])
    validations["oracle_same_transform_for_paired_cases"] = bool(np.allclose(
        matrices[(CASES[0], "oracle_stable", -1)], matrices[(CASES[1], "oracle_stable", -1)], rtol=0, atol=1e-12))
    validations["original_sources_unchanged"] = all(sha256(Path(path))==value for path, value in input_hashes.items())
    save_csv(out / "all_runs.csv", all_rows)
    dump(out / "all_runs.json", all_rows)
    summaries = []
    metric_keys = ["rotation_error_deg", "translation_error_mm", "stable_holdout_pose_rmse_mm",
                   "stable_holdout_noisy_rmse_mm", "all_point_nn_after_rmse_mm", "fit_elapsed_seconds"]
    for case in CASES:
        for method in METHODS:
            group = [row for row in all_rows if row["case"]==case and row["method"]==method]
            summary = {"case": case, "method": method, "runs": len(group),
                       "fit_reference_points":group[0]["fit_reference_points"], "fit_moving_points":group[0]["fit_moving_points"]}
            for key in metric_keys:
                values = np.array([row[key] for row in group])
                summary.update({f"{key}_mean":float(values.mean()), f"{key}_std":float(values.std(ddof=1)) if len(values)>1 else 0.0,
                                f"{key}_min":float(values.min()), f"{key}_max":float(values.max()),
                                f"{key}_p05":float(np.quantile(values,.05)), f"{key}_p95":float(np.quantile(values,.95))})
            summaries.append(summary)
    dump(out / "summary.json", summaries)
    save_csv(out / "summary.csv", summaries)
    selected = {(row["case"], row["method"]):row for row in summaries}
    paired_effects = []
    for method in METHODS:
        a = {row["repeat_index"]:row for row in all_rows if row["case"]==CASES[0] and row["method"]==method}
        b = {row["repeat_index"]:row for row in all_rows if row["case"]==CASES[1] and row["method"]==method}
        differences = np.array([b[index]["translation_error_mm"]-a[index]["translation_error_mm"] for index in a])
        check_differences = np.array([b[index]["stable_holdout_pose_rmse_mm"]-a[index]["stable_holdout_pose_rmse_mm"] for index in a])
        paired_effects.append({"method":method,"paired_runs":len(differences),
            "deposition_minus_rigid_translation_error_mm_mean":float(differences.mean()),
            "deposition_minus_rigid_translation_error_mm_std":float(differences.std(ddof=1)) if len(differences)>1 else 0.0,
            "deposition_minus_rigid_stable_pose_error_mm_mean":float(check_differences.mean())})
    save_csv(out / "paired_change_effect.csv", paired_effects)
    case = CASES[1]
    ordinary = selected[(case,"all_points")]
    oracle = selected[(case,"oracle_stable")]
    random = selected[(case,"random_matched")]
    improvement = (1-oracle["translation_error_mm_mean"]/ordinary["translation_error_mm_mean"])*100
    hypothesis_supported = oracle["translation_error_mm_mean"] < ordinary["translation_error_mm_mean"] and oracle["stable_holdout_pose_rmse_mm_mean"] < ordinary["stable_holdout_pose_rmse_mm_mean"] and oracle["translation_error_mm_mean"] < random["translation_error_mm_p05"]
    finding = {
        "hypothesis": "On this fixed paired dataset, local changed points participating in pose fitting cause bias; excluding the true change footprint can reduce that bias beyond count reduction alone",
        "supported_on_this_dataset": bool(hypothesis_supported),
        "deposition_oracle_translation_reduction_percent":improvement,
        "oracle_translation_below_random_empirical_p05":bool(oracle["translation_error_mm_mean"] < random["translation_error_mm_p05"]),
        "random_draws":args.repeats,
        "limits":"One terrain patch, one prescribed change and one data/noise seed. Random-subset repeats measure sampling variability only; empirical quantiles are not population confidence intervals. Region exclusion also changes geometry distribution; this test does not isolate every spatial/normal effect.",
        "not_an_automatic_method":"Oracle reads true labels; all methods share ordinary point-to-point ICP. This experiment does not evaluate py4dgeo stable-area ICP, and is not an innovation claim.",
        "next_step":"Design input-only candidate-stable-region selection and compare with mature trimmed/robust ICP and spatial/normal sampling controls; do not use truth masks or held-out checks in fitting or parameter selection.",
    }
    dump(out / "findings.json", finding)
    dump(out / "validation.json", {"passed":all(validations.values()),"checks":validations})
    # Fixed-axis comparison within each metric; random error bars show draw SD.
    fig, axes = plt.subplots(2,2,figsize=(10.5,7.2),layout="constrained")
    colors=["#6a90bf","#299f72","#d89b4b"]
    for row_index, case in enumerate(CASES):
        for col_index,(key,label) in enumerate((("rotation_error_deg","Rotation error (deg)"),("translation_error_mm","Translation error (mm)"))):
            ax=axes[row_index,col_index]
            means=[selected[(case,m)][f"{key}_mean"] for m in METHODS]
            std=[selected[(case,m)][f"{key}_std"] for m in METHODS]
            bars=ax.bar(["All points","Oracle stable",f"Random (n={args.repeats})"],means,yerr=std,capsize=4,color=colors)
            ax.set_ylabel(label);ax.set_title(case);ax.set_ylim(0,max(np.array(means)+np.array(std))*1.3)
            for b,v,s in zip(bars,means,std):ax.text(b.get_x()+b.get_width()/2,v+s+max(means)*.03,f"{v:.4f}",ha="center",fontsize=9)
            ax.spines[["top","right"]].set_visible(False)
    fig.savefig(out/"diagnosis_comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,case in zip(axes,CASES):
        values=[r["translation_error_mm"] for r in all_rows if r["case"]==case and r["method"]=="random_matched"]
        ax.hist(values,bins=8,color=colors[2],alpha=.8,label="Matched random draws")
        ax.axvline(selected[(case,"all_points")]["translation_error_mm_mean"],color=colors[0],label="All points")
        ax.axvline(selected[(case,"oracle_stable")]["translation_error_mm_mean"],color=colors[1],label="Oracle stable")
        ax.set_title(case);ax.set_xlabel("Translation error (mm)");ax.set_ylabel("Number of draws");ax.legend(fontsize=8)
    fig.savefig(out/"random_control_distribution.png",dpi=180);plt.close(fig)
    with np.load(dataset/CASES[1]/"truth/ground_truth.npz") as saved:
        check_xy=saved["stable_check_target"][:,:2]
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.4),layout="constrained")
    maximum=max(errors[(CASES[1],method)].max() for method in ("all_points","oracle_stable"))
    for ax,method in zip(axes,("all_points","oracle_stable")):
        im=ax.scatter(check_xy[:,0],check_xy[:,1],c=errors[(CASES[1],method)],s=9,vmin=0,vmax=maximum,cmap="viridis")
        ax.set_title(method);ax.set_xlabel("Local X (m)");ax.set_ylabel("Local Y (m)");ax.set_aspect("equal")
    fig.colorbar(im,ax=axes,label="Independent stable check pose error (mm)")
    fig.savefig(out/"deposition_stable_check_errors.png",dpi=180);plt.close(fig)
    lines=["# 稳定参考区域的作用：受控诊断实验", "", "本次对应待办清单第4步：定位已有方法的问题和可改进空间。没有实现自动稳定区识别，也没有在这里调用 py4dgeo 的稳定区域 ICP。", "",
        "## 实验设计", "", f"固定使用 `{args.dataset}` 的两组输入。所有拟合均调用同一套 NumPy Kabsch + SciPy 精确 KDTree 的普通点到点 ICP，从单位矩阵开始，最多150次迭代，RMSE变化阈值1e-9米。没有使用真值匹配关系或独立检查点拟合。", "",
        "| 方法 | 拟合使用的点 | 真值使用范围 |", "| --- | --- | --- |", "| all_points 普通 ICP | 全部参考点和待配准点 | 只在拟合结束后评价 |",
        f"| oracle_stable 理想稳定区 ICP | {len(oracle_ref):,}个参考点、{len(oracle_mov):,}个待配准点；排除半径{exclusion_radius:g}米的已知变化位置及边界缓冲区 | 使用真值筛选，是理想对照 |",
        f"| random_matched 等数量随机对照 | 分别从全部两期点中均匀随机选择与理想对照相同数量的点；固定{args.repeats}个种子 | 只从理想对照取得点数，不按变化标签选点 |", "",
        "纯刚性组虽然不存在真实变化，也排除同一潜在变化位置，保持两组的几何选择一致。两组使用相同的随机采样索引。随机点数匹配控制了拟合点数量，但没有完全控制区域分布和法向方向分布，解释应保留这个限制。", "",
        "## 结果", "", "随机对照显示均值 ± 样本标准差，反映同一数据上的随机选点波动，不是跨地形/噪声重复实验的置信区间。", "",
        "| 测试组 | 方法 | 旋转误差/度 | 平移误差/mm | 独立稳定区位姿残差/mm |", "| --- | --- | ---: | ---: | ---: |"]
    names={"all_points":"普通 ICP（全部点）","oracle_stable":"理想稳定区 ICP（真值筛选）","random_matched":f"等数量随机 ICP（{args.repeats}次）"}
    case_names={CASES[0]:"第1组：纯配准",CASES[1]:"第2组：局部堆积"}
    for case in CASES:
        for method in METHODS:
            s=selected[(case,method)]
            def value(key):
                return f"{s[key+'_mean']:.4f} ± {s[key+'_std']:.4f}" if method=="random_matched" else f"{s[key+'_mean']:.4f}"
            lines.append(f"| {case_names[case]} | {names[method]} | {value('rotation_error_deg')} | {value('translation_error_mm')} | {value('stable_holdout_pose_rmse_mm')} |")
    lines += ["", "配对比较同一选点方案由无堆积到有堆积的平移误差增量：", "",
              "| 方法 | 平移误差增量/mm |", "| --- | ---: |"]
    for effect in paired_effects:
        text=f"{effect['deposition_minus_rigid_translation_error_mm_mean']:.4f}"
        if effect["paired_runs"]>1:
            text+=f" ± {effect['deposition_minus_rigid_translation_error_mm_std']:.4f}"
        lines.append(f"| {names[effect['method']]} | {text} |")
    lines += ["", "![位姿误差对照](diagnosis_comparison.png)", "", "![独立稳定检查点误差](deposition_stable_check_errors.png)", "",
        "## 解释", "", f"在局部堆积组中，理想筛选使平移误差相对全部点{'降低' if improvement>=0 else '增加'}约 {abs(improvement):.2f}%。本次预先设定的诊断条件（平移和稳定检查区误差均下降，且理想对照平移误差低于随机对照的经验5%分位）{'满足' if hypothesis_supported else '未全部满足'}。", "",
        "理想对照两组的输入选点完全相同且均未发生堆积，因而应得到相同变换；代码已核对。全部点输出也重现了上一步的基线矩阵。", "",
        f"这支持的范围仅是：在本固定地形、此处合成堆积和本数据随机种子下，准确排除变化位置可以减少普通 ICP 的位姿偏差。它不证明自动稳定区识别已经实现，也不证明某种新算法普遍优于现有方法。{args.repeats}次随机采样并不能代替多个数据集和变化配置。", "",
        "全点最近邻 RMSE 作为补充指标保存于CSV；局部堆积应当保留，不能用全部区域距离越小越好评价。体积估计器仍是后续步骤，本次没有把真值体积当作算法估计体积。", "",
        "## 下一步的可检验假设", "", "主要问题：真实变化区参与刚性位姿拟合，会将部分局部变化吸收到整体变换中，使未变化区域也出现偏差。", "",
        "待验证假设：从粗配准后的两期输入中识别局部变化候选区域，减少其参与拟合，可以降低独立稳定检查区残差，同时保留局部堆积。正式筛选机制只能使用输入点云，不能读取本次真值、正确变换或留出检查点。", "",
        "实现前还需与成熟的截尾/稳健 ICP 以及空间/法向采样方法核对，明确新增步骤；不能把普通残差剔除或真值掩膜称为创新。后续固定当前数据，再用更多噪声和变化配置检验泛化。", "",
        "## 查看与复现", "", "- 每组 all_points/after_overlay.ply：普通 ICP 后的完整两期点云，蓝=t1、橙=t2。",
        "- 每组 oracle_stable/after_overlay.ply：理想筛选估计变换后，应用到完整 t2 的效果；变化区没有从输出点云删除。",
        "- 每组 random_matched/seed_00/after_overlay.ply：预先指定第0次随机抽样，未挑选最好结果。",
        "- oracle_reference_selection.ply：绿色是参与理想拟合的参考点，红色是排除区。",
        "- oracle_moving_selection_truth_aligned.ply：仅为展示而用真值对齐的待配准点筛选图，不参与拟合。",
        "- 每次试验有 transformation.csv、fit_selection.npz、metrics.json 和 iterations.csv。",
        "- all_runs.csv/json 保留所有结果；summary.csv/json 汇总；manifest.json 记录协议及输入哈希；validation.json 记录验证。", "",
        "CloudCompare 文件使用原基准数据的局部坐标系；原点记录在 manifest.json。两期应同时采用相同显示方式查看，不要与原始大坐标母本混合叠加。", "",
        "```powershell", f"& .\\.venv\\Scripts\\python.exe scripts\\diagnose_stable_region.py --repeats {args.repeats} --seed {args.seed}", "```", "",
        f"完整验证状态：{'通过' if all(validations.values()) else '未全部通过，需检查 validation.json'}。"]
    (out/"诊断报告.md").write_text("\n".join(lines),encoding="utf-8")
    LOG.info("Diagnosis supported on current dataset: %s; validation passed: %s",hypothesis_supported,all(validations.values()))
    LOG.info("Deposition translation error all/oracle/random mean: %.4f / %.4f / %.4f mm",
             ordinary["translation_error_mm_mean"],oracle["translation_error_mm_mean"],random["translation_error_mm_mean"])
    if not all(validations.values()):
        raise RuntimeError("Export/experimental validation failed; inspect validation.json")


if __name__=="__main__":
    main()
