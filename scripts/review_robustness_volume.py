"""Review persisted grids and report limits; never refit or tune registration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def review(out):
    rows=json.loads((out/"all_results.json").read_text(encoding="utf-8"))
    overall=json.loads((out/"overall_summary.json").read_text(encoding="utf-8"))
    summaries=json.loads((out/"scenario_summary.json").read_text(encoding="utf-8"))
    checks={}
    coverage={}
    # Normalize coverage using the same quadrature for numerator/denominator.
    # Comparing discrete cells to an analytic integral can otherwise give
    # 100.00035% for a completely covered smooth bump due to quadrature error.
    for name in dict.fromkeys(row["case"] for row in rows):
        folder=out/"evaluation"/name
        with np.load(folder/"volume_grids.npz") as saved:
            common=saved["common_valid"]
            delta=saved["true_delta_z_evaluation_only"]
            step=float(saved["cell_size_m"])
            full_mass=float(np.abs(delta).sum()*step**2)
            common_mass=float(np.abs(delta[common]).sum()*step**2)
            ratio=common_mass/full_mass if full_mass else None
            coverage[name]=ratio
            for row in (r for r in rows if r["case"]==name):
                value=float((saved[f"{row['method']}_z"][common]-saved["reference_z"][common]).sum()*step**2)
                checks[f"{name}_{row['method']}_saved_grid_volume"]=bool(np.isclose(value,row["estimated_common_net_m3"],rtol=0,atol=1e-10))
                row["true_volume_signal_coverage_fraction"]=ratio
            checks[f"{name}_coverage_in_range"]=ratio is None or 0<=ratio<=1+1e-12
        record=json.loads((folder/"coverage.json").read_text(encoding="utf-8"))
        record["volume_signal_coverage_fraction"]=ratio
        record["coverage_denominator"]="Same-grid integral of absolute true change over complete planned ROI"
        dump(folder/"coverage.json",record)
    dump(out/"all_results.json",rows)
    with (out/"all_results.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lookup={r["method"]:r for r in overall}
    candidate=lookup["patch_candidate"]
    base=lookup["ordinary"]
    huber=lookup["huber"]
    point=lookup["pointwise_gate"]
    new=[row for row in rows if row["split"]=="new_conditions"]
    cases=list(dict.fromkeys(row["case"] for row in new))
    matched={(row["case"],row["method"]):row for row in new}
    wins={method:sum(matched[(case,"patch_candidate")]["net_absolute_error_m3"]<matched[(case,method)]["net_absolute_error_m3"] for case in cases)
          for method in ("ordinary","huber","pointwise_gate")}
    ratios=[coverage[name] for name in cases if coverage[name] is not None]
    findings={"new_cases":len(cases),"candidate_volume_MAE_reduction_vs_ordinary_percent":(1-candidate["mean_net_volume_absolute_error_m3"]/base["mean_net_volume_absolute_error_m3"])*100,
        "candidate_volume_MAE_reduction_vs_huber_percent":(1-candidate["mean_net_volume_absolute_error_m3"]/huber["mean_net_volume_absolute_error_m3"])*100,
        "candidate_beats_pointwise_ablation_overall":candidate["mean_net_volume_absolute_error_m3"]<point["mean_net_volume_absolute_error_m3"],
        "candidate_per_case_volume_wins":wins,"true_change_signal_coverage_min":min(ratios),"true_change_signal_coverage_max":max(ratios),
        "main_limit":"Regional aggregation has no overall advantage over the pointwise ablation in these tests; sparse sampling and interpolation remain important error sources. One terrain only.",
        "correct_pose_is_not_a_lower_bound":"Correct-pose volume results isolate estimator effects; biased poses can sometimes cancel interpolation/noise bias by chance."}
    dump(out/"findings.json",findings)
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    protocol["coverage_fraction_definition"]="Numerator and denominator use the same fixed-grid absolute-change quadrature; full analytic volume remains a separate field"
    protocol["evaluation_code_hashes_final"]={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        (ROOT/"scripts/run_robustness_volume.py",ROOT/"scripts/terrain_volume.py",Path(__file__))}
    dump(out/"protocol.json",protocol)
    names={"rigid_control":"纯配准","deposition_nominal":"常规堆积","deposition_noise15mm":"噪声增至15 mm",
           "deposition_sparse":"待配准点云变稀","deposition_low":"低幅堆积0.2 m","excavation":"局部挖除","deposition_broad":"大范围堆积"}
    lines=["# 本阶段结论与适用边界", "", "算法和参数保持冻结，仅增加噪声、抽样及局部变化配置。以下结论包含全部预设条件，不只选择有利案例。", "",
        f"1. 候选区域筛选相对普通ICP的平均净体积绝对误差降低约 **{findings['candidate_volume_MAE_reduction_vs_ordinary_percent']:.1f}%**。",
        f"2. 候选的体积误差均值比Huber约低 **{findings['candidate_volume_MAE_reduction_vs_huber_percent']:.1f}%**，但位姿误差仍高于Huber，且不是每例更好。",
        "3. **逐点法向残差筛选消融的总体体积误差更低，尚不能证明区域聚合带来总体新增收益。** 需要保留这个负面消融结果。", "",
        "| 方法 | 平移误差均值/mm | 净体积绝对误差均值/m³ |", "| --- | ---: | ---: |"]
    for method,label in (("ordinary","普通ICP"),("huber","Huber加权ICP"),("pointwise_gate","逐点筛选消融"),("patch_candidate","区域筛选候选"),("oracle_pose","正确变换体积对照")):
        r=lookup[method];lines.append(f"| {label} | {r['mean_translation_error_mm']:.4f} | {r['mean_net_volume_absolute_error_m3']:.4f} |")
    lines += ["", "## 哪些条件有收益，哪些条件存在问题", "", "| 条件 | Huber体积误差均值/m³ | 逐点筛选/m³ | 区域候选/m³ |", "| --- | ---: | ---: | ---: |"]
    for scenario in dict.fromkeys(row["scenario"] for row in summaries):
        values=[next(row["net_absolute_error_m3_mean"] for row in summaries if row["scenario"]==scenario and row["method"]==method) for method in ("huber","pointwise_gate","patch_candidate")]
        lines.append(f"| {names[scenario]} | {' | '.join(f'{value:.4f}' for value in values)} |")
    lines += ["", "较大范围堆积下区域候选的结果优于逐点筛选，是一个值得进一步检验的现象；当前只有一个地形和3个噪声种子，不据此认定已证明普遍优势。变稀条件下候选误差明显增大，应作为适用边界保留。", "",
        "正确变换对照在变稀条件下也出现较大体积偏差，说明独立采样和高程插值会影响结果；实际算法的体积绝对误差偶尔小于正确变换对照，可能是配准偏差与插值偏差发生抵消，不能据此声称位姿更正确。", "",
        f"新增非零变化案例的已知体积信号覆盖比例为 {min(ratios)*100:.3f}%～{max(ratios)*100:.3f}%。该比例用同一网格积分归一化，避免解析与离散积分微差造成超过100%。工作区空洞仍未当作零，所有主表误差都在相同公共覆盖上计算。", "",
        "## 对课程报告的影响", "", "目前可以支持的结论是：抑制真实变化对配准的影响，能改善体积估计；自动筛选初版可复现。当前证据不足以把区域聚合写成已经优于成熟方法和简化消融的新算法。", "",
        "下一步应先确定报告采用的主方案及其有限适用条件，再整理方法流程、对照和消融结果。若继续主张区域机制的技术贡献，需要额外的独立地形或预先设计的复杂空间变化验证，而不是在这批结果上反复调参数。", "",
        "本次保留所有脚本、参数、数据、逐例结果及失败边界，便于如实撰写课程报告。"]
    (out/"结论与适用边界.md").write_text("\n".join(lines),encoding="utf-8")
    report=out/"鲁棒性与体积实验报告.md"
    content=report.read_text(encoding="utf-8").split("\n## 结果解读与边界\n")[0]
    content=content.replace("误差底限，不能当成实际算法","估计误差基线（并非严格下界），不能当成实际算法")
    report.write_text(content+"\n\n## 结果解读与边界\n\n详见 [结论与适用边界.md](结论与适用边界.md)，包含总体排名、变稀失败情况、消融结论及覆盖归一化说明。\n",encoding="utf-8")
    dump(out/"review_validation.json",{"passed":all(checks.values()),"checks":checks})
    if not all(checks.values()):raise RuntimeError("Saved-grid review failed")
    print(json.dumps(findings,ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="robustness_volume_v1")
    args=parser.parse_args()
    review(ROOT/"results"/args.output)
