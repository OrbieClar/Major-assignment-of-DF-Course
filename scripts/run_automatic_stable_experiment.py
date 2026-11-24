"""Run input-only algorithms first; open truth files only for separate evaluation."""
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from automatic_stable_icp import METHODS
from build_terrain_benchmark import apply, overlay, write_ply, read_ascii_ply
from diagnose_stable_region import evaluate

ROOT=Path(__file__).resolve().parents[1]
CASES=("01_rigid_only","02_local_deposition")
SCREENERS=("pointwise_gate","patch_no_dilation","patch_candidate")
NAMES={"ordinary":"普通 ICP", "trimmed_90":"固定90%截尾 ICP", "huber":"Huber 加权 ICP",
       "normal_space":"法向空间采样 ICP", "pointwise_gate":"逐点法向残差筛选（消融）",
       "patch_no_dilation":"区域筛选、不扩张（消融）", "patch_candidate":"区域一致性筛选候选方法"}


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default="terrain_benchmark_v1")
    parser.add_argument("--output",default="automatic_stable_v1")
    parser.add_argument("--config",default="configs/automatic_stable_v1.json")
    args=parser.parse_args()
    if any(Path(name).name!=name for name in (args.dataset,args.output)):
        parser.error("Use simple dataset/output folder names")
    dataset=ROOT/"results"/args.dataset
    out=ROOT/"results"/args.output
    out.mkdir(parents=True,exist_ok=True)
    logs=ROOT/"logs"
    logs.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[
        logging.StreamHandler(sys.stdout),logging.FileHandler(logs/f"{args.output}.log",encoding="utf-8")])
    config_path=ROOT/args.config
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if config["coarse_iterations"]+config["refine_iterations"]>config["max_iterations"]:
        raise ValueError("Candidate iteration budget exceeds control budget")
    # Freeze a byte-identical parameter snapshot before fitting or evaluating any method.
    frozen=out/"frozen_config.json"
    frozen.write_bytes(config_path.read_bytes())
    config_hash=digest(frozen)
    inputs=[dataset/case/"inputs.npz" for case in CASES]
    input_hashes={str(p):digest(p) for p in inputs}
    code_hashes={str(p):digest(p) for p in (Path(__file__),ROOT/"scripts/automatic_stable_icp.py")}
    protocol={"dataset":str(dataset),"frozen_config_sha256":config_hash,"input_hashes":input_hashes,
        "code_hashes":code_hashes,"truth_read_order":"Both input-only worker processes must finish before evaluation opens any truth files",
        "development_status":"Single existing paired terrain dataset used for development/evaluation; not a held-out dataset benchmark",
        "solver":"Exact KDTree + proper weighted Kabsch, all methods initialized at identity",
        "fairness":"150 ICP update cap; candidate/ablations allocate at most 50 coarse + 100 refine. Feature time is charged to each relevant method. Sampling/weights/objectives differ as specified.",
        "parameter_choice":"Common constants fixed before this run; thresholds adapt only to observed spacing/residuals. No known change center, radius, mask, height, transformation or generation noise is passed to workers.",
        "python":sys.executable,"methods":list(METHODS)}
    dump(out/"protocol.json",protocol)
    # Finish all algorithm execution before the evaluator opens a truth or metadata file.
    for case in CASES:
        log_path=logs/f"{args.output}_{case}_worker.log"
        command=[sys.executable,"-u",str(ROOT/"scripts/automatic_stable_icp.py"),
                 "--input",str(dataset/case/"inputs.npz"),"--output",str(out/case),
                 "--config",str(frozen),"--dataset-root",str(dataset)]
        logging.info("Input-only worker: %s",case)
        with log_path.open("w",encoding="utf-8") as logfile:
            result=subprocess.run(command,stdout=logfile,stderr=subprocess.STDOUT,timeout=180,cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"Worker failed: {case}; inspect {log_path}")
    # Separate evaluation stage. Truth cannot influence already persisted algorithm outputs.
    metadata=json.loads((dataset/"metadata.json").read_text(encoding="utf-8"))
    dump(out/"metadata.json",{"global_origin":metadata["global_origin"],"coordinate_units":"meters",
        "formula":"exported_source_xyz = local_xyz + global_origin","source_dataset":str(dataset)})
    rows=[]
    validation={}
    diagnostics={}
    for case in CASES:
        with np.load(dataset/case/"inputs.npz") as saved:
            reference,moving=saved["reference"].copy(),saved["moving"].copy()
        with np.load(dataset/case/"truth/ground_truth.npz") as saved:
            truth={key:saved[key].copy() for key in saved.files}
        audit=json.loads((out/case/"input_access_audit.json").read_text(encoding="utf-8"))
        validation[f"{case}_input_only_audit"]=audit["only_coordinate_input_read"]
        diagnostics[case]=json.loads((out/case/"diagnostics.json").read_text(encoding="utf-8"))
        for method in METHODS:
            folder=out/case/method
            with np.load(folder/"algorithm_result.npz") as saved:
                matrix,ref_mask,mov_mask=saved["transformation"].copy(),saved["reference_keep_mask"].copy(),saved["moving_keep_mask"].copy()
            info=json.loads((folder/"algorithm_info.json").read_text(encoding="utf-8"))
            metrics,aligned,_=evaluate(reference,moving,matrix,truth)
            changed=truth["moving_changed_mask"]
            rejected=~mov_mask
            tp=int(np.sum(rejected & changed))
            fp=int(np.sum(rejected & ~changed))
            fn=int(np.sum(mov_mask & changed))
            is_screener=method in SCREENERS
            row={"case":case,"method":method,"method_label":NAMES[method],
                 "retained_reference_points":int(ref_mask.sum()),"retained_moving_points":int(mov_mask.sum()),
                 "icp_iterations":info["fit_iterations"],"elapsed_seconds":info["elapsed_seconds"],
                 **metrics,"rejected_change_precision":tp/(tp+fp) if is_screener and tp+fp else None,
                 "rejected_change_recall":tp/(tp+fn) if is_screener and tp+fn else None,
                 "rejected_unchanged_fraction":fp/max(1,int((~changed).sum())) if is_screener else None,
                 "retained_change_points":fn if is_screener else None,
                 "fallback":info["selection_info"]["fallback"]}
            rows.append(row)
            dump(folder/"evaluation.json",row)
            np.savetxt(folder/"transformation.csv",matrix,delimiter=",",fmt="%.16g")
            np.savetxt(folder/"moving_aligned.xyz",aligned,fmt="%.12f")
            overlay(folder/"after_overlay.ply",reference,aligned)
            if is_screener or method=="normal_space":
                write_ply(folder/"reference_selection.ply",reference,
                    np.where(ref_mask[:,None],np.array([45,180,95]),np.array([225,65,55])),
                    {"selected_for_fit":ref_mask.astype(float)})
                write_ply(folder/"moving_selection_auto_aligned.ply",aligned,
                    np.where(mov_mask[:,None],np.array([45,180,95]),np.array([225,65,55])),
                    {"selected_for_fit":mov_mask.astype(float)})
            loaded,_=read_ascii_ply(folder/"after_overlay.ply")
            validation[f"{case}_{method}_ply_roundtrip"]=bool(np.allclose(loaded,np.vstack((reference,aligned)),rtol=0,atol=1e-10))
            validation[f"{case}_{method}_proper_rotation"]=bool(np.isclose(np.linalg.det(matrix[:3,:3]),1,atol=1e-8,rtol=0) and np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-8,rtol=0))
            validation[f"{case}_{method}_within_iteration_budget"]=info["fit_iterations"]<=config["max_iterations"]
            restored=np.loadtxt(folder/"transformation.csv",delimiter=",")
            validation[f"{case}_{method}_matrix_roundtrip"]=bool(np.allclose(apply(moving,restored),np.loadtxt(folder/"moving_aligned.xyz"),rtol=0,atol=1e-10))
            if method=="ordinary":
                previous=np.loadtxt(dataset/case/"ordinary_icp_transform.csv",delimiter=",")
                validation[f"{case}_ordinary_matches_verified_baseline"]=bool(np.allclose(matrix,previous,rtol=0,atol=1e-9))
            logging.info("%s %s: translation %.4f mm; stable pose %.4f mm; kept %d / %d",case,method,
                row["translation_error_mm"],row["stable_holdout_pose_rmse_mm"],int(ref_mask.sum()),int(mov_mask.sum()))
    validation["inputs_unchanged"]=all(digest(Path(p))==h for p,h in input_hashes.items())
    validation["frozen_parameters_unchanged"]=digest(frozen)==config_hash and digest(config_path)==config_hash
    dump(out/"validation.json",{"passed":all(validation.values()),"checks":validation})
    dump(out/"results.json",rows)
    with (out/"results.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    selected={(row["case"],row["method"]):row for row in rows}
    changes=selected[(CASES[1],"patch_candidate")]
    baseline=selected[(CASES[1],"ordinary")]
    controls=[selected[(CASES[1],method)] for method in ("trimmed_90","huber","normal_space")]
    best=min(controls,key=lambda row:row["translation_error_mm"])
    findings={"candidate_translation_improvement_over_ordinary_percent":(1-changes["translation_error_mm"]/baseline["translation_error_mm"])*100,
        "candidate_beats_best_established_control_translation":changes["translation_error_mm"]<best["translation_error_mm"],
        "best_established_control_by_translation":best["method"],"candidate_translation_error_mm":changes["translation_error_mm"],
        "best_control_translation_error_mm":best["translation_error_mm"],
        "interpretation":"All first-version results retained. No parameter search using known pose or masks. Current dataset is a development case, not evidence of broad superiority or patent novelty.",
        "pending":"No py4dgeo/Yang full-method reproduction in this experiment; no volume estimator; robustness across independent terrain/noise/change settings remains to be tested."}
    dump(out/"findings.json",findings)
    labels=["Ordinary","Trim 90%","Huber","Normal-space","Point gate","Patch no grow","Patch candidate"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    colors=["#8a9fb8"]*4+["#c4ac86","#bbaac8","#229579"]
    for ax,case in zip(axes,CASES):
        values=[selected[(case,m)]["translation_error_mm"] for m in METHODS]
        bars=ax.barh(labels,values,color=colors);ax.invert_yaxis();ax.set_title(case)
        ax.set_xlabel("Translation error (mm)");ax.set_xlim(0,max(values)*1.25)
        for bar,value in zip(bars,values):ax.text(value+max(values)*.02,bar.get_y()+bar.get_height()/2,f"{value:.3f}",va="center",fontsize=9)
    fig.savefig(out/"method_comparison.png",dpi=180);plt.close(fig)
    with np.load(out/CASES[1]/"patch_candidate/algorithm_result.npz") as saved:
        mask=saved["reference_keep_mask"]
    with np.load(dataset/CASES[1]/"inputs.npz") as saved:
        ref=saved["reference"].copy()
    fig,ax=plt.subplots(figsize=(6,5),layout="constrained")
    ax.scatter(ref[mask,0],ref[mask,1],s=2,c="#30a36d",label="Selected")
    ax.scatter(ref[~mask,0],ref[~mask,1],s=4,c="#dc554a",label="Rejected")
    ax.set_aspect("equal");ax.set_xlabel("Local X (m)");ax.set_ylabel("Local Y (m)");ax.legend()
    ax.set_title("Input-only patch selection (deposition case)")
    fig.savefig(out/"automatic_selection.png",dpi=180);plt.close(fig)
    lines=["# 自动稳定区域筛选：初版实现与对照", "", "本阶段实现只读取两期坐标的候选筛选机制，同时保留成熟对照与消融。它是开发初版，不是已经证实有新颖性的算法。", "",
        "## 算法是否读取答案", "", "算法在两个独立 worker 中先执行完毕，然后评价脚本才读取真值。worker 的审计钩子禁止读取数据集内除指定 inputs.npz 外的任何文件；每组 input_access_audit.json 保存访问记录。候选算法未获得堆积圆心、半径、高度、生成噪声、正确位姿或标签。真值掩膜仅用于拟合后评价识别精度。", "",
        "候选参数见 frozen_config.json，运行前固定并记录SHA256；尺度由参考点云中位最近邻间距估计。两组采用同一参数配置。当前两组是已使用过的开发数据，不属于独立留出的测试集。", "",
        "## 候选流程", "", "```mermaid", "flowchart TD", ' A["S1 两期坐标与局部法向"] --> B["S2 普通 ICP 粗配准"]',
        ' B --> C["S3 双向法向残差与稳健阈值"]', ' C --> D["S4 网格残差中位数、同号比例、连通块"]',
        ' D --> E["S5 扩张疑似变化单元并筛选参考点"]', ' E --> F{"保留点数量足够？"}',
        ' F -->|是| G["S6 在保留区域精配准"]', ' F -->|否| H["返回粗配准并记录回退"]', "```", "",
        "法向通过24邻点PCA计算，朝+Z统一方向，适用于当前地形高程曲面假设。粗配准上限50步；候选与两个消融的精配准上限100步，总更新次数上限150。最终拟合仍为正确 Kabsch 的点到点 ICP，法向只用于筛选证据。", "",
        "双向匹配的法向残差减去全局中位数，以1.4826×MAD估计尺度，尺度下界为点间距的0.05倍；阈值是该尺度的3倍。网格边长取点间距的6倍。单元至少12个双向配对观测，中位残差超阈值且同号超过半阈值的比例至少0.65，才标为疑似变化。保留同号26邻接中至少3个单元组成的连通块，扩张一层网格，再从两期剔除对应区域及距离过大的不可靠匹配。若任一期保留不足50%或30点则回退。双向观测并非独立样本，不将这些比例解释为统计置信度。", "",
        "## 对照与消融的实际实现", "", "| 方法 | 实际实现 |", "| --- | --- |",
        "| 普通 ICP | 全部点，最多150次更新 |", "| 固定90%截尾 ICP | 每次保留欧氏最近邻残差最小的90%点对；是固定比例对照，不是完整的自适应重叠率 TrICP 复现 |",
        "| Huber 加权 ICP | 以欧氏点对距离的MAD、点间距下界和1.345倍系数构造权重，用加权Kabsch迭代；不冒充Open3D点到面稳健ICP实现 |",
        "| 法向空间采样 ICP | +Z半球按方位和余弦分层，每期至多6000点，再普通ICP；是法向采样思想的明确实现，不是原论文全套设置 |",
        "| 逐点残差筛选 | 同样粗配准、法向残差和阈值，逐点筛选，不使用区域连通与扩张 |",
        "| 区域筛选、不扩张 | 与候选相同，关闭网格扩张，检查边界扩张作用 |",
        "| 区域一致性候选 | 完整S1—S6流程 |", "",
        "所有方法从单位矩阵开始，最多150次ICP更新；有的包含特征预处理，不能把迭代次数相同理解为运行成本相同。耗时包括各自需要的法向/筛选成本，公共缓存时间也对相关方法计费。", "",
        "## 实际结果", "", "| 组别 | 方法 | 平移误差/mm | 旋转误差/度 | 独立稳定区位姿残差/mm | 保留t2点数 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for case in CASES:
        for method in METHODS:
            row=selected[(case,method)]
            case_name="纯配准" if case==CASES[0] else "局部堆积"
            lines.append(f"| {case_name} | {NAMES[method]} | {row['translation_error_mm']:.4f} | {row['rotation_error_deg']:.6f} | {row['stable_holdout_pose_rmse_mm']:.4f} | {row['retained_moving_points']} |")
    lines += ["", "![各方法位姿误差](method_comparison.png)", "", "![候选筛选区域](automatic_selection.png)", "",
        f"局部堆积组中，候选相对普通ICP的平移误差变化（正数为改善）为 {findings['candidate_translation_improvement_over_ordinary_percent']:.2f}%。成熟对照中平移误差最低的是 {NAMES[best['method']]}（{best['translation_error_mm']:.4f} mm）；候选{'优于' if findings['candidate_beats_best_established_control_translation'] else '没有优于'}这个对照。所有方法及两个消融均保留，没有按测试真值挑选参数后隐藏失败设置。", "",
        "## 自动筛选评价", "", "只对三个明确输出稳定区域掩膜的方法计算变化区识别指标；截尾、加权和法向采样的保留点不当作变化检测标签。变化点真值包含隆起边缘极小正高差，识别率不能代替位姿评价。", "",
        "| 方法（局部堆积组） | 排除点中的真实变化比例 | 真实变化点排除率 | 未变化点误排比例 |", "| --- | ---: | ---: | ---: |"]
    for method in SCREENERS:
        row=selected[(CASES[1],method)]
        values=["—" if row[k] is None else f"{row[k]*100:.2f}%" for k in ("rejected_change_precision","rejected_change_recall","rejected_unchanged_fraction")]
        lines.append(f"| {NAMES[method]} | {' | '.join(values)} |")
    lines += ["", "## 已有工作与区别边界", "", "详见同目录已有方法核对.md。残差剔除、MAD、法向空间采样、区域稳定性和稳健加权均有成熟来源。本初版组合双向法向残差、单元同号一致性、连通块和扩张，属于候选工程流程；尚未证明它相对最接近方法有独立技术增量。不能据此直接宣称专利新颖性。", "",
        "本次未把此前异常的py4dgeo 1.2.0稳定区ICP重新包装成成功结果，也没有完整复现Yang/Schwieger的超体素方法。该已有方法对照仍应单列为后续待核查项。", "",
        "## 文件与复现", "", "- configs/automatic_stable_v1.json：工程参数；frozen_config.json：本次冻结参数。",
        "- scripts/automatic_stable_icp.py：仅坐标算法及受审计worker；scripts/run_automatic_stable_experiment.py：运行和独立评价。",
        "- 每组每方法的 after_overlay.ply：参考t1蓝色、变换后的完整t2橙色；变化点没有从最终输出删除。",
        "- patch_candidate/reference_selection.ply：自动选中的点绿色、排除点红色；moving_selection_auto_aligned.ply 是应用估计变换后的t2筛选图。",
        "- algorithm_result.npz、algorithm_info.json、iterations.csv：算法原始输出；evaluation.json：事后真值评价。",
        "- results.csv/json、findings.json、validation.json、protocol.json：完整结果、解释、验证和协议。",
        "- input_only_evidence.npz、patch_statistics.csv、diagnostics.json：粗配准残差、网格统计及估计尺度。",
        "- CloudCompare文件沿用terrain_benchmark的局部米制坐标，原点见metadata.json。", "",
        "```powershell", f"& .\\.venv\\Scripts\\python.exe scripts\\run_automatic_stable_experiment.py --dataset {args.dataset} --output {args.output}", "```", "",
        "当前仅一个地形、一个噪声种子和一个堆积配置。冻结此初版后，后续应使用独立噪声/密度/变化配置测试，并实现体积估计；不要根据本次真值反复调整阈值再报告为独立测试。", "",
        f"验证：{'全部通过' if all(validation.values()) else '有未通过项，见validation.json'}。"]
    (out/"自动筛选实验报告.md").write_text("\n".join(lines),encoding="utf-8")
    logging.info("Finished; validation %s; candidate vs best established control: %s",all(validation.values()),findings["candidate_beats_best_established_control_translation"])
    if not all(validation.values()):
        raise RuntimeError("Validation failed; inspect validation.json")


if __name__=="__main__":
    main()
