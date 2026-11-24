"""Frozen-algorithm perturbation study plus common-domain 2.5D volume evaluation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automatic_stable_icp import METHODS, transform
from build_terrain_benchmark import read_ascii_ply, write_ply, overlay, bump
from diagnose_stable_region import evaluate
from terrain_volume import interpolate_height, integrate, analytical_checks

ROOT=Path(__file__).resolve().parents[1]
LABELS={"ordinary":"普通ICP","trimmed_90":"截尾ICP","huber":"Huber加权ICP","normal_space":"法向采样ICP",
        "pointwise_gate":"逐点筛选消融","patch_no_dilation":"不扩张消融","patch_candidate":"区域筛选候选","oracle_pose":"正确变换对照"}


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_save(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def create_case(folder,master,origin,center,scenario,seed,config):
    truth_folder=folder/"truth"
    truth_folder.mkdir(parents=True,exist_ok=True)
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    sampling,rng_ref,rng_mov,rng_check,rng_order=streams
    holdout=sampling.random(len(master))<config["holdout_fraction"]
    ru,mu=sampling.random(len(master)),sampling.random(len(master))
    ref_ids=np.flatnonzero(~holdout & (ru<scenario["reference_keep"]))
    mov_ids=np.flatnonzero(~holdout & (mu<scenario["moving_keep"]))
    ref_ids=rng_order.permutation(ref_ids);mov_ids=rng_order.permutation(mov_ids)
    # Generate full-length noise fields, so density variants keep the same noise
    # for every shared source vertex, not a different prefix of random values.
    reference_noise=rng_ref.normal(size=master.shape)*scenario["noise_sigma_m"]
    moving_noise=rng_mov.normal(size=master.shape)*scenario["noise_sigma_m"]
    reference=master[ref_ids]+reference_noise[ref_ids]
    dz=bump(master[mov_ids,:2],center,scenario["radius_m"],scenario["height_m"])
    clean=master[mov_ids].copy();clean[:,2]+=dz
    noisy=clean+moving_noise[mov_ids]
    forward=np.eye(4)
    forward[:3,:3]=Rotation.from_euler("xyz",config["forward_angles_xyz_deg"],degrees=True).as_matrix()
    forward[:3,3]=config["forward_translation_m"]
    correct=np.linalg.inv(forward)
    moving=transform(noisy,forward)
    # Hold out the union of all planned possible change footprints for every
    # profile, making stable checkpoint geometry comparable within each seed.
    check_radius=max(s["radius_m"] for s in config["scenarios"])+.75
    check_ids=np.flatnonzero(holdout & (np.linalg.norm(master[:,:2]-center,axis=1)>check_radius))
    target=master[check_ids]
    check_noise=rng_check.normal(size=master.shape)*scenario["noise_sigma_m"]
    np.savez_compressed(folder/"inputs.npz",reference=reference,moving=moving)
    np.savez_compressed(truth_folder/"ground_truth.npz",correct_registration=correct,forward_transform=forward,
        moving_changed_mask=abs(dz)>0,moving_true_dz=dz,moving_true_aligned_noisy=noisy,moving_true_aligned_noiseless=clean,
        stable_check_target=target,stable_check_moving_noiseless=transform(target,forward),
        stable_check_moving_noisy=transform(target+check_noise[check_ids],forward),
        reference_source_indices=ref_ids,moving_source_indices=mov_ids,stable_check_source_indices=check_ids)
    truth={"scenario":scenario,"seed":seed,"global_origin":origin.tolist(),"center_local_xy_m":center.tolist(),
           "analytic_net_volume_m3":float(np.pi*scenario["radius_m"]**2*scenario["height_m"]/3),
           "changed_moving_points":int(np.sum(abs(dz)>0)),"reference_points":len(reference),"moving_points":len(moving),
           "stable_holdout_points":len(target),"model":"Clean piecewise-linear mother height field plus prescribed compact vertical polynomial bump; independent noisy vertex subsets",
           "truth_access":"Evaluation only; input-only workers cannot read this folder"}
    dump(truth_folder/"truth.json",truth)
    if seed==config["seeds"][0]:
        write_ply(folder/"reference.ply",reference,(40,135,235))
        write_ply(folder/"moving.ply",moving,(245,130,35))
    assert not np.intersect1d(check_ids,np.r_[ref_ids,mov_ids]).size
    assert np.allclose(transform(moving,correct),noisy,rtol=0,atol=1e-10)
    return truth


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default="configs/robustness_volume_v1.json")
    parser.add_argument("--output",default="robustness_volume_v1")
    args=parser.parse_args()
    if Path(args.output).name!=args.output:parser.error("Use a simple output folder name")
    out=ROOT/"results"/args.output
    out.mkdir(parents=True,exist_ok=True)
    logs=ROOT/"logs";logs.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[
        logging.StreamHandler(sys.stdout),logging.FileHandler(logs/f"{args.output}.log",encoding="utf-8")])
    config_path=ROOT/args.config
    config=json.loads(config_path.read_text(encoding="utf-8"))
    algorithm_path=ROOT/config["algorithm_config"]
    frozen_algorithm=out/"frozen_algorithm_config.json"
    frozen_algorithm.write_bytes(algorithm_path.read_bytes())
    (out/"frozen_study_config.json").write_bytes(config_path.read_bytes())
    locked={str(p):sha(p) for p in (algorithm_path,config_path,ROOT/"scripts/automatic_stable_icp.py",ROOT/config["source"])}
    # The original experiment is kept intact and also evaluated for volume.
    original=ROOT/"results/terrain_benchmark_v1"
    original_meta=json.loads((original/"metadata.json").read_text(encoding="utf-8"))
    previous=ROOT/"results/automatic_stable_v1"
    assert sha(frozen_algorithm)==sha(previous/"frozen_config.json"),"Algorithm parameters changed since the initial development experiment"
    global_master,_=read_ascii_ply(ROOT/config["source"])
    assert sha(ROOT/config["source"])==original_meta["source_sha256"],"Mother geometry changed"
    origin=np.asarray(original_meta["global_origin"])
    master=global_master-origin
    center=np.asarray(original_meta["bump"]["center_local_xy_m"])
    hull=ConvexHull(master[:,:2])
    clearance=float(np.min(-(hull.equations[:,:2]@center+hull.equations[:,2])))
    assert clearance>max(s["radius_m"] for s in config["scenarios"]),"A planned support disk exceeds the modeled surface hull"
    step=config["grid_step_m"]
    minimum=master[:,:2].min(0)+config["reference_extent_margin_m"]
    maximum=master[:,:2].max(0)-config["reference_extent_margin_m"]
    axes=[np.arange(lo+step/2,hi,step) for lo,hi in zip(minimum,maximum)]
    gx,gy=np.meshgrid(*axes,indexing="xy")
    grid=np.column_stack((gx.ravel(),gy.ravel()))
    validations=analytical_checks()
    check_booleans={key:value for key,value in validations.items() if isinstance(value,bool)}
    dump(out/"volume_analytical_tests.json",validations)
    protocol={"study":"Frozen algorithm, one terrain, seven profiles and three new paired sampling/noise seeds",
        "locks":locked,"algorithm_config_sha256":sha(frozen_algorithm),"grid_step_m":step,
        "grid_origin_xy_m":minimum.tolist(),"grid_shape":list(gx.shape),"grid_cells":len(grid),
        "global_origin":origin.tolist(),"ROI":"Fixed mother bounding box inset by 0.75 m, selected without using the change mask",
        "height_estimator":"Delaunay linear interpolation of each observed cloud to cell centers; reject outside hull, long triangles and distant XY support",
        "integration":"Each method and correct-pose control share the intersection of supported cells for that case; no invalid cell is treated as zero",
        "truth_comparison":"Error is against known change integrated on that same common support. Full analytical volume and signal coverage are reported separately",
        "oracle_pose":"Correct transformation applied only in evaluation to quantify interpolation/sampling/noise effects; never supplied to registration workers",
        "development_vs_new":"Original two cases summarized separately; new seeds/configurations are perturbation tests on the same terrain, not independent geographical validation"}
    dump(out/"protocol.json",protocol)
    dump(out/"metadata.json",{"global_origin":origin.tolist(),"coordinate_units":"m","formula":"source_xyz=local_xyz+global_origin"})
    cases=[]
    datasets=out/"datasets";datasets.mkdir(exist_ok=True)
    algorithm_outputs=out/"algorithms";algorithm_outputs.mkdir(exist_ok=True)
    for scenario in config["scenarios"]:
        for seed in config["seeds"]:
            name=f"{scenario['name']}_seed{seed}"
            folder=datasets/name
            truth=create_case(folder,master,origin,center,scenario,seed,config)
            cases.append({"name":name,"scenario":scenario["name"],"seed":seed,"split":"new_conditions",
                          "data":folder,"algorithm":algorithm_outputs/name,"truth":truth,"profile":scenario})
    dump(out/"case_index.json",[{key:(str(value) if isinstance(value,Path) else value) for key,value in case.items()} for case in cases])
    # Complete every input-only worker before volume/pose evaluation begins.
    for index,case in enumerate(cases):
        logging.info("Fitting %d/%d %s",index+1,len(cases),case["name"])
        with (logs/f"{args.output}_{case['name']}_worker.log").open("w",encoding="utf-8") as log:
            result=subprocess.run([sys.executable,"-u",str(ROOT/"scripts/automatic_stable_icp.py"),
                "--input",str(case["data"]/"inputs.npz"),"--output",str(case["algorithm"]),
                "--config",str(frozen_algorithm),"--dataset-root",str(datasets)],
                stdout=log,stderr=subprocess.STDOUT,timeout=180,cwd=ROOT)
        if result.returncode:raise RuntimeError(f"Input-only worker failed: {case['name']}")
        audit=json.loads((case["algorithm"]/"input_access_audit.json").read_text(encoding="utf-8"))
        check_booleans[f"{case['name']}_input_only"]=audit["only_coordinate_input_read"]
    # Reuse, do not overwrite, the two existing development cases.
    for name,height in (("01_rigid_only",0.),("02_local_deposition",.5)):
        profile={"name":name,"radius_m":4.5,"height_m":height,"noise_sigma_m":.005,"reference_keep":.9,"moving_keep":.9}
        cases.append({"name":name,"scenario":name,"seed":20261123,"split":"development",
                      "data":original/name,"algorithm":previous/name,"profile":profile})
    rows=[]
    evaluations=out/"evaluation";evaluations.mkdir(exist_ok=True)
    for index,case in enumerate(cases):
        logging.info("Volume/pose evaluation %d/%d %s",index+1,len(cases),case["name"])
        folder=evaluations/case["name"];folder.mkdir(exist_ok=True)
        with np.load(case["data"]/"inputs.npz") as saved:
            reference,moving=saved["reference"].copy(),saved["moving"].copy()
        with np.load(case["data"]/"truth/ground_truth.npz") as saved:
            truth={key:saved[key].copy() for key in saved.files}
        ref_z,ref_valid=interpolate_height(reference,grid,config["maximum_xy_support_distance_m"],config["maximum_triangle_edge_m"])
        matrices={}
        surfaces={}
        masks={}
        for method in METHODS:
            with np.load(case["algorithm"]/method/"algorithm_result.npz") as saved:
                matrices[method]=saved["transformation"].copy()
        matrices["oracle_pose"]=truth["correct_registration"]
        for method,matrix in matrices.items():
            aligned=transform(moving,matrix)
            surfaces[method],masks[method]=interpolate_height(aligned,grid,config["maximum_xy_support_distance_m"],config["maximum_triangle_edge_m"])
        common=ref_valid.copy()
        for valid in masks.values():common &= valid
        if common.sum()<100:raise RuntimeError("Insufficient common supported area")
        delta_true=bump(grid,center,case["profile"]["radius_m"],case["profile"]["height_m"])
        truth_volume=integrate(delta_true,common,step)
        full_analytic=float(np.pi*case["profile"]["radius_m"]**2*case["profile"]["height_m"]/3)
        full_grid=float(delta_true.sum()*step**2)
        full_signal=float(np.abs(delta_true).sum()*step**2)
        common_signal=float(np.abs(delta_true[common]).sum()*step**2)
        check_booleans[f"{case['name']}_full_analytic_vs_grid"]=abs(full_grid-full_analytic)<.005
        oracle_volume=integrate(surfaces["oracle_pose"]-ref_z,common,step)
        grid_arrays={"xy":grid,"grid_shape":gx.shape,"cell_size_m":step,"reference_z":ref_z,
                     "reference_valid":ref_valid,"common_valid":common,"true_delta_z_evaluation_only":delta_true}
        for method,matrix in matrices.items():
            metrics,aligned,_=evaluate(reference,moving,matrix,truth)
            volumes=integrate(surfaces[method]-ref_z,common,step)
            error=volumes["net_m3"]-truth_volume["net_m3"]
            row={"split":case["split"],"case":case["name"],"scenario":case["scenario"],"seed":case["seed"],"method":method,
                 "noise_sigma_m":case["profile"]["noise_sigma_m"],"moving_keep_probability":case["profile"]["moving_keep"],
                 "radius_m":case["profile"]["radius_m"],"peak_height_m":case["profile"]["height_m"],
                 **metrics,"true_analytic_full_net_m3":full_analytic,"true_common_net_m3":truth_volume["net_m3"],
                 "estimated_common_net_m3":volumes["net_m3"],"net_absolute_error_m3":abs(error),"net_signed_error_m3":error,
                 "net_relative_error_percent":abs(error)/abs(truth_volume["net_m3"])*100 if abs(truth_volume["net_m3"])>1e-12 else None,
                 "true_common_added_m3":truth_volume["added_m3"],"true_common_removed_m3":truth_volume["removed_m3"],
                 "estimated_common_added_m3":volumes["added_m3"],"estimated_common_removed_m3":volumes["removed_m3"],
                 "pose_effect_on_estimated_net_m3":volumes["net_m3"]-oracle_volume["net_m3"],
                 "correct_pose_volume_error_m3":oracle_volume["net_m3"]-truth_volume["net_m3"],
                 "common_area_m2":truth_volume["covered_area_m2"],"common_area_fraction_of_ROI":float(common.mean()),
                 "individual_surface_coverage_fraction":float((ref_valid & masks[method]).mean()),
                 "true_volume_signal_coverage_fraction":common_signal/full_signal if full_signal>0 else None}
            rows.append(row)
            grid_arrays[f"{method}_z"]=surfaces[method]
            grid_arrays[f"{method}_valid"]=masks[method]
            if method in ("ordinary","huber","patch_candidate","oracle_pose") and (case["split"]=="development" or case["seed"]==config["seeds"][0]):
                overlay(folder/f"{method}_after_overlay.ply",reference,aligned)
                display_z=(ref_z[common]+surfaces[method][common])/2
                difference=surfaces[method][common]-ref_z[common]
                color=(plt.get_cmap("coolwarm")(np.clip(difference,-.6,.6)/1.2+.5)[:,:3]*255).astype(np.uint8)
                write_ply(folder/f"{method}_height_difference.ply",np.column_stack((grid[common],display_z)),color,
                          {"estimated_delta_z_m":difference,"true_delta_z_m_evaluation":delta_true[common]})
            check_booleans[f"{case['name']}_{method}_rigid"]=bool(np.isclose(np.linalg.det(matrix[:3,:3]),1,rtol=0,atol=1e-8))
        np.savez_compressed(folder/"volume_grids.npz",**grid_arrays)
        dump(folder/"coverage.json",{"common_cells":int(common.sum()),"ROI_cells":len(grid),"cell_area_m2":step**2,
            "common_area_m2":truth_volume["covered_area_m2"],"full_analytic_net_volume_m3":full_analytic,
            "true_common_net_volume_m3":truth_volume["net_m3"],
            "volume_signal_coverage_fraction":common_signal/full_signal if full_signal>0 else None,
            "note":"All method errors use common coverage truth, not the full analytical volume; missing area is not zero change"})
    csv_save(out/"all_results.csv",rows);dump(out/"all_results.json",rows)
    summaries=[]
    metrics=("translation_error_mm","rotation_error_deg","stable_holdout_pose_rmse_mm","net_absolute_error_m3",
             "net_signed_error_m3","pose_effect_on_estimated_net_m3","correct_pose_volume_error_m3","common_area_fraction_of_ROI")
    for scenario in config["scenarios"]:
        for method in (*METHODS,"oracle_pose"):
            group=[row for row in rows if row["split"]=="new_conditions" and row["scenario"]==scenario["name"] and row["method"]==method]
            item={"scenario":scenario["name"],"method":method,"runs":len(group)}
            for key in metrics:
                values=np.array([row[key] for row in group])
                item[f"{key}_mean"]=float(values.mean());item[f"{key}_std"]=float(values.std(ddof=1))
            summaries.append(item)
    csv_save(out/"scenario_summary.csv",summaries);dump(out/"scenario_summary.json",summaries)
    overall=[]
    for method in (*METHODS,"oracle_pose"):
        group=[row for row in rows if row["split"]=="new_conditions" and row["method"]==method]
        overall.append({"method":method,"cases":len(group),
            "mean_translation_error_mm":float(np.mean([row["translation_error_mm"] for row in group])),
            "mean_stable_pose_rmse_mm":float(np.mean([row["stable_holdout_pose_rmse_mm"] for row in group])),
            "mean_net_volume_absolute_error_m3":float(np.mean([row["net_absolute_error_m3"] for row in group])),
            "median_net_volume_absolute_error_m3":float(np.median([row["net_absolute_error_m3"] for row in group]))})
    csv_save(out/"overall_summary.csv",overall);dump(out/"overall_summary.json",overall)
    check_booleans["locked_source_algorithm_and_configs_unchanged"]=all(sha(Path(p))==value for p,value in locked.items())
    # Re-integrate actual saved grid files, independently checking report serialization.
    grid_file=evaluations/"02_local_deposition/volume_grids.npz"
    with np.load(grid_file) as saved:
        for method in (*METHODS,"oracle_pose"):
            value=integrate(saved[f"{method}_z"]-saved["reference_z"],saved["common_valid"],float(saved["cell_size_m"]))["net_m3"]
            recorded=next(row["estimated_common_net_m3"] for row in rows if row["case"]=="02_local_deposition" and row["method"]==method)
            check_booleans[f"{method}_saved_volume_reintegrates"]=bool(np.isclose(value,recorded,rtol=0,atol=1e-10))
    dump(out/"validation.json",{"passed":all(check_booleans.values()),"checks":check_booleans})
    relevant=("ordinary","trimmed_90","huber","pointwise_gate","patch_candidate")
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.5),layout="constrained")
    for ax,key,label in zip(axes,("mean_translation_error_mm","mean_net_volume_absolute_error_m3"),("Mean translation error (mm)","Mean net-volume absolute error (m³)")):
        values=[next(row[key] for row in overall if row["method"]==m) for m in relevant]
        bars=ax.barh(["Ordinary","Trim 90%","Huber","Point gate","Patch candidate"],values,color=["#91a4bb"]*3+["#c5b58c","#249e7b"])
        ax.invert_yaxis();ax.set_xlabel(label);ax.set_xlim(0,max(values)*1.25)
        for bar,value in zip(bars,values):ax.text(value+max(values)*.02,bar.get_y()+bar.get_height()/2,f"{value:.3f}",va="center")
    fig.suptitle("21 new perturbation cases on one terrain; fixed algorithm parameters")
    fig.savefig(out/"robustness_summary.png",dpi=180);plt.close(fig)
    # Input-only height differences and true changes are shown on identical support.
    with np.load(grid_file) as saved:
        mask=saved["common_valid"];xy=saved["xy"][mask]
        fig,axes=plt.subplots(1,4,figsize=(13,3.7),layout="constrained")
        fields=[saved["true_delta_z_evaluation_only"][mask]]+[saved[f"{m}_z"][mask]-saved["reference_z"][mask] for m in ("ordinary","huber","patch_candidate")]
        for ax,title,field in zip(axes,("Known clean change","Ordinary","Huber","Patch candidate"),fields):
            image=ax.scatter(xy[:,0],xy[:,1],c=field,s=2,cmap="coolwarm",vmin=-.5,vmax=.5)
            ax.set_aspect("equal");ax.set_title(title);ax.set_xlabel("Local X (m)")
        fig.colorbar(image,ax=axes,label="Vertical change (m)")
        fig.savefig(out/"development_height_differences.png",dpi=180);plt.close(fig)
    lines=["# 固定参数鲁棒性与体积估计实验", "", "新增7种条件×3个独立抽样/噪声种子，共21个半合成案例；另对原先两组开发数据补算体积。仍使用同一个AHK地形，不属于跨地形或外部数据集验证。全部既有算法和参数保持不变。", "",
        "## 条件", "", "| 条件 | 每轴噪声/mm | t2保留概率 | 半径/m | 峰值高差/m |", "| --- | ---: | ---: | ---: | ---: |"]
    for s in config["scenarios"]:lines.append(f"| {s['name']} | {s['noise_sigma_m']*1000:g} | {s['moving_keep']:.2f} | {s['radius_m']:g} | {s['height_m']:g} |")
    lines += ["", "同一个种子内共享母本、留出点、抽样随机场和标准噪声场，用于单因素配对比较；密度变体保留的共同点具有相同噪声。粗配准误差仍为XYZ角度[0.4,-0.3,1.0]度、平移[0.3,-0.2,0.15]米，尚未测试更大初始误差。", "",
        "## 体积估计口径", "", "采用XY三角网线性插值，在统一0.25米方格的中心估计两期高程。超出凸包、距离观测点超过0.75米、所在三角形最大边长超过2米的单元无效。对每个案例，所有方法及正确变换对照使用共同有效单元的交集；不把无数据区域当零。", "",
        "`净变化量 = sum((z_t2-z_t1) * 0.25²)`；分别另存正高差堆积量和负高差挖除量。该单元体积公式与CloudCompare的2.5D体积相同，但本实现使用散点三角网插值，不是逐项复刻CloudCompare的格内平均/插值设置。参考：[CloudCompare官方说明](https://cloudcompare.org/doc/wiki/index.php/2.5D_Volume)。", "",
        "**主表误差比较的是共同覆盖范围内的已知体积。** 全圆解析体积也单独保存，但覆盖缺失时不能直接与它比较。每例coverage.json给出有效面积、解析体积、覆盖内真值和体积信号覆盖比例。ROI从母本边界内缩0.75米固定，不按变化掩膜裁剪；真值只在算法执行结束后评价。", "",
        "正确变换对照（oracle_pose）直接使用已知逆变换，专门估计噪声、独立采样和插值带来的估计误差基线（并非严格下界），不能当成实际算法。`pose_effect_on_estimated_net_m3` 是实际算法体积与该对照之差；它不等于总真值误差。", "",
        "原始正负高差积分未额外应用检测限，正负体积会累积噪声和插值起伏；因此主比较使用带符号净变化量，并保留两种分量供检查。悬挑和多值高程不适用本2.5D模型。", "",
        "## 新增21例总体结果", "", "下表是绝对误差的均值，不是对有正负的误差先平均而相互抵消。不同条件的体积和覆盖范围不同，均值只描述本预设实验集合。", "",
        "| 方法 | 平移误差均值/mm | 独立稳定区残差均值/mm | 净体积绝对误差均值/m³ |", "| --- | ---: | ---: | ---: |"]
    for row in overall:lines.append(f"| {LABELS[row['method']]} | {row['mean_translation_error_mm']:.4f} | {row['mean_stable_pose_rmse_mm']:.4f} | {row['mean_net_volume_absolute_error_m3']:.4f} |")
    lines += ["", "![总体结果](robustness_summary.png)", "", "## 每种条件的体积误差", "", "均值 ± 样本标准差，每种条件3个种子；这不是跨地形置信区间。", "",
              "| 条件 | 普通ICP/m³ | Huber/m³ | 区域筛选候选/m³ | 正确变换对照/m³ |", "| --- | ---: | ---: | ---: | ---: |"]
    for scenario in config["scenarios"]:
        values=[]
        for method in ("ordinary","huber","patch_candidate","oracle_pose"):
            item=next(s for s in summaries if s["scenario"]==scenario["name"] and s["method"]==method)
            values.append(f"{item['net_absolute_error_m3_mean']:.4f} ± {item['net_absolute_error_m3_std']:.4f}")
        lines.append(f"| {scenario['name']} | {' | '.join(values)} |")
    lines += ["", "![原开发堆积组高差图](development_height_differences.png)", "",
        "## 文件与使用", "", "- all_results.csv/json：所有方法、所有案例，包括覆盖、位姿和正负体积。",
        "- scenario_summary.csv/json：各条件3个种子的均值和标准差；overall_summary.csv/json：新案例总体汇总。",
        "- datasets/：新增输入及独立truth目录；algorithms/：审计通过的仅坐标算法输出。",
        "- evaluation/<案例>/volume_grids.npz：统一坐标网格、各方法高程、有效掩膜和评价用真值。",
        "- evaluation/<案例>/*_height_difference.ply：以两期平均高程作为展示位置，高差保存在estimated_delta_z_m标量场；这不是把Z坐标本身改成高差。",
        "- evaluation/<案例>/*_after_overlay.ply：参考蓝色、完整配准后t2橙色。为控制文件量，只对原开发组及每个条件的第一个新种子导出PLY。",
        "- 配置、输入来源和代码哈希见protocol.json；解析平面测试与回读检查见validation.json。", "",
        "```powershell", "& .\\.venv\\Scripts\\python.exe scripts\\run_robustness_volume.py", "```", "",
        f"全部验证：{'通过' if all(check_booleans.values()) else '有失败项，请检查validation.json'}。后续结论应依据全部条件及成熟对照，不从本测试集合反复调整参数。"]
    (out/"鲁棒性与体积实验报告.md").write_text("\n".join(lines),encoding="utf-8")
    from review_robustness_volume import review
    review(out)
    logging.info("Finished %d method-case results; validation %s",len(rows),all(check_booleans.values()))
    for row in overall:logging.info("%s: mean translation %.4f mm, mean volume abs error %.4f m3",row["method"],row["mean_translation_error_mm"],row["mean_net_volume_absolute_error_m3"])
    if not all(check_booleans.values()):raise RuntimeError("Validation failed")


if __name__=="__main__":main()
