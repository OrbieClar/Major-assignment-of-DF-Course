"""Reproducible report figures from frozen results; no algorithm retuning.

The portable bundle also accepts --source-data source_data --output .
All coordinates and quantitative panels use saved observations, not mock data.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent / "figure_helpers"))
from audit_panel_alignment import require_matplotlib_panel_alignment

PROJECT = Path(__file__).resolve().parents[1]
CASE = "deposition_nominal_seed202611241"
METHODS = ["ordinary", "trimmed_90", "huber", "normal_space", "pointwise_gate",
           "patch_no_dilation", "patch_candidate", "oracle_pose"]
LABELS = dict(zip(METHODS, ["普通ICP", "90%截尾", "Huber加权", "法向空间采样", "逐点筛选（主方案）",
                           "区域筛选、不扩张", "区域筛选、扩张", "正确位姿对照"]))
COLORS = {"ordinary":"#9CA3AF", "trimmed_90":"#B7C4CD", "huber":"#65798C",
          "normal_space":"#C2CBD0", "pointwise_gate":"#2379B8", "patch_no_dilation":"#D8BDA4",
          "patch_candidate":"#C98845", "oracle_pose":"#303A45"}
SCENARIOS = ["rigid_control", "deposition_nominal", "deposition_noise15mm", "deposition_sparse",
             "deposition_low", "excavation", "deposition_broad"]
SCENARIO_LABELS = ["纯刚性", "标准堆积", "15 mm\n噪声", "待测点云\n变稀", "低幅堆积", "挖除", "大范围\n堆积"]
mpl.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Microsoft YaHei","DejaVu Sans"],
    "font.size":7.5, "axes.titlesize":8, "axes.labelsize":7.5, "xtick.labelsize":7,
    "ytick.labelsize":7, "legend.fontsize":7, "axes.spines.top":False, "axes.spines.right":False,
    "axes.linewidth":.65, "axes.unicode_minus":False, "svg.fonttype":"none", "pdf.fonttype":42,
    "savefig.dpi":600, "figure.facecolor":"white", "axes.facecolor":"white"})


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f); writer.writerow(columns); writer.writerows(rows)


def prepare_sources(target):
    """Materialize figure-only input and evaluation files without changing results."""
    target.mkdir(parents=True,exist_ok=True)
    base=PROJECT/"results/robustness_volume_v1"
    shutil.copy2(base/"all_results.csv",target/"all_results.csv")
    shutil.copy2(base/"evaluation"/CASE/"volume_grids.npz",target/"height_grids.npz")
    config=json.loads((base/"frozen_algorithm_config.json").read_text(encoding="utf-8-sig"))
    with np.load(base/"datasets"/CASE/"inputs.npz") as z:
        reference,moving=z["reference"],z["moving"]
    with np.load(base/"algorithms"/CASE/"pointwise_gate/algorithm_result.npz") as z:
        result={k:z[k] for k in z.files}
    with np.load(base/"datasets"/CASE/"truth/ground_truth.npz") as z:
        changed=z["moving_changed_mask"]
    # Recompute only the input-derived coarse residuals needed to illustrate
    # the saved gate. No true pose or true change mask enters these operations.
    sys.path.insert(0,str(PROJECT/"scripts"))
    from automatic_stable_icp import icp, estimate_normals, geometry_evidence, transform
    from scipy.spatial import cKDTree
    spacing=float(np.median(cKDTree(reference).query(reference,k=2)[0][:,1]))
    normals=estimate_normals(reference,config["normal_neighbors"])
    coarse,_=icp(reference,moving,config,iterations=config["coarse_iterations"])
    evidence=geometry_evidence(reference,moving,coarse,normals,spacing,config)
    ref_mask=evidence["valid_r"] & (abs(evidence["residual_r"]-evidence["location"])<=evidence["threshold"])
    mov_mask=evidence["valid_m"] & (abs(evidence["residual_m"]-evidence["location"])<=evidence["threshold"])
    assert np.array_equal(ref_mask,result["reference_keep_mask"])
    assert np.array_equal(mov_mask,result["moving_keep_mask"])
    info=json.loads((base/"algorithms"/CASE/"pointwise_gate/algorithm_info.json").read_text(encoding="utf-8-sig"))
    np.savez_compressed(target/"representative_points.npz",reference=reference,moving=moving,
        aligned_moving=transform(moving,result["transformation"]),transformation=result["transformation"],
        reference_keep_mask=ref_mask,moving_keep_mask=mov_mask,
        moving_changed_evaluation_only=changed,forward_centered_residual_m=evidence["residual_m"]-evidence["location"],
        reverse_centered_residual_m=evidence["residual_r"]-evidence["location"],
        forward_valid=evidence["valid_m"],reverse_valid=evidence["valid_r"],
        threshold_m=evidence["threshold"],spacing_m=spacing)
    (target/"representative_metadata.json").write_text(json.dumps({"case":CASE,
        "representative_selection":"first predeclared seed of nominal deposition; not selected by outcome",
        "reference_points":len(reference),"moving_points":len(moving),"known_changed_points":int(changed.sum()),
        "saved_masks_recomputed_exactly":True,"algorithm_info":info},ensure_ascii=False,indent=2),encoding="utf-8")


def panel(ax, letter, title):
    ax.annotate(letter,xy=(0,1),xycoords="axes fraction",xytext=(-17,8),
                textcoords="offset points",ha="left",va="bottom",weight="bold",fontsize=9)
    ax.set_title(title,loc="left",pad=8)
    ax.tick_params(direction="out",length=2.5,width=.6)


def heading(fig, text, subtitle=None):
    fig.text(.05,.975,text,fontsize=10,weight="bold",va="top")
    if subtitle:
        fig.text(.05,.935,subtitle,fontsize=7.3,color="#546171",va="top")


def spatial(ax, bounds, xlabel=True, ylabel=True):
    ax.set_xlim(bounds[0],bounds[1]); ax.set_ylim(bounds[2],bounds[3])
    ax.set_box_aspect(1)
    ax.set_xticks([-20,0,20]); ax.set_yticks([-20,0,20])
    if xlabel: ax.set_xlabel("局部X / m")
    if ylabel: ax.set_ylabel("局部Y / m")


def export(fig, stem, output, axes=None, exclude_axes=()):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,exclude_axes=exclude_axes,
        json_out=output/"qa"/(stem+".alignment.json"),
        overlay_svg=output/"qa"/(stem+".alignment.svg"),
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=(len(axes or [])>1),strict=True)
    renderer=fig.canvas.get_renderer()
    clipping=[]
    # Matplotlib keeps extra rounded ticks outside the axis limits as visible
    # artists even though they are not drawn. Audit only the rendered ticks.
    tick_texts=set(); rendered_tick_texts=set()
    for axis in fig.axes:
        for coordinate in (axis.xaxis,axis.yaxis):
            for tick in coordinate.get_major_ticks()+coordinate.get_minor_ticks():
                tick_texts.update((id(tick.label1),id(tick.label2)))
            if axis.axison and coordinate.get_visible():
                for tick in coordinate._update_ticks():
                    rendered_tick_texts.update((id(tick.label1),id(tick.label2)))
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text().strip():continue
        if id(artist) in tick_texts and id(artist) not in rendered_tick_texts:continue
        bb=artist.get_window_extent(renderer)
        if bb.x0 < -.5 or bb.y0 < -.5 or bb.x1 > fig.bbox.width+.5 or bb.y1 > fig.bbox.height+.5:
            clipping.append({"text":artist.get_text(),"bbox_px":list(bb.bounds)})
    (output/"qa"/(stem+".canvas_text_bounds.json")).write_text(json.dumps(
        {"passed":not clipping,"outside_canvas":clipping},ensure_ascii=False,indent=2),encoding="utf-8")
    assert not clipping,(stem,clipping)
    fig.savefig(output/(stem+".pdf"),format="pdf",facecolor="white")
    fig.savefig(output/(stem+".svg"),format="svg",facecolor="white")
    fig.savefig(output/(stem+".png"),dpi=600,facecolor="white")
    fig.savefig(output/(stem+".tiff"),dpi=600,facecolor="white",pil_kwargs={"compression":"tiff_lzw"})
    plt.close(fig)


def flow(output):
    fig,ax=plt.subplots(figsize=(7.20472440945,8.26771653543))
    fig.subplots_adjust(left=.04,right=.96,bottom=.02,top=.94)
    ax.set(xlim=(0,10),ylim=(0,14.4)); ax.axis("off")
    heading(fig,"图1  法向残差筛选配准与体积估计流程")
    def box(x,y,text,w=8.3,h=.83,fill="#F0F5FA"):
        ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=.04,rounding_size=.08",
                                   fc=fill,ec="#66788A",lw=.75))
        ax.text(x,y,text,ha="center",va="center",fontsize=8.2,linespacing=1.45)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=9,lw=.8,color="#425466"))
    ys=[13.3,12.0,10.7,9.4,8.1]
    texts=["S01  两期点云输入与坐标整理\n统一单位与局部原点，配准方向P→Q",
           "S02  点间距与局部法向估计\n参考点最近邻间距中位数s；24点PCA，法向统一+Z",
           "S03  普通ICP初步配准\n最多50次，得到初步变换T0",
           "S04  双向法向残差与稳健阈值\n有效对应d≤3s；中位数μ、MAD尺度σ；τ=3σ",
           "S05  两期拟合参考点分别筛选\n保留有效对应且 |r−μ|≤τ 的点；只筛选一次"]
    for y,t in zip(ys,texts): box(5,y,t)
    for hi,lo in zip(ys,ys[1:]): arrow((5,hi-.46),(5,lo+.46))
    ax.add_patch(Polygon([(5,7.16),(8.5,6.48),(5,5.80),(1.5,6.48)],fc="#FBF3DF",ec="#9B803A",lw=.75))
    arrow((5,7.65),(5,7.18))
    ax.text(5,6.48,"两期各保留≥50%\n且各≥30点？",ha="center",va="center",fontsize=8.2)
    box(2.55,4.85,"S06a  筛选点ICP精配准\n以T0为初值，最多100次",w=4.3,fill="#E8F2F9")
    box(7.45,4.85,"S06b  返回初步变换\nT=T0，恢复全点掩膜并记录回退",w=4.3,fill="#FAEEE3")
    arrow((3.5,6.18),(2.55,5.32)); arrow((6.5,6.18),(7.45,5.32))
    ax.text(2.8,5.9,"是",fontsize=8);ax.text(7.2,5.9,"否",fontsize=8)
    box(5,3.34,"S07  完整待测点云变换与导出\n最终变换作用于全部点，保留真实堆积和挖除",fill="#E8F2F9")
    arrow((2.55,4.38),(4,3.80));arrow((7.45,4.38),(6,3.80))
    box(5,1.90,"S08  高差与体积估计\n完整两期TIN插值，0.25 m网格；无支持单元不填零")
    arrow((5,2.88),(5,2.37))
    ax.text(5,.83,"筛选仅用于估计位姿；体积计算使用完整点云。",ha="center",fontsize=8,color="#425466")
    ax.text(5,.32,"前提：初始对齐足够接近、有重叠稳定地形、满足2.5D高程模型。",ha="center",fontsize=7.5,color="#425466")
    export(fig,"Fig01_method_flow",output,axes=[ax])


def selection(output, data, meta):
    fig,axs=plt.subplots(2,2,figsize=(7.20472440945,6.81102362205))
    fig.subplots_adjust(left=.10,right=.94,bottom=.10,top=.85,wspace=.80,hspace=.65)
    heading(fig,"图2  候选稳定参考点的筛选与独立核对","标准堆积，预设种子1；真实地形母本上的半合成变化")
    a,b,c,d=axs.ravel(); ref=data["reference"]; mov=data["aligned_moving"]
    bounds=(-23,23,-23,23)
    panel(a,"a","参考地形与局部起伏")
    image=a.scatter(ref[:,0],ref[:,1],c=ref[:,2],s=.65,cmap="cividis",rasterized=True,linewidths=0)
    spatial(a,bounds)
    fig.canvas.draw(); pos=a.get_position()
    ca=fig.add_axes([pos.x1+.018,pos.y0,.012,pos.height])
    colorbar=fig.colorbar(image,cax=ca);colorbar.set_label("局部高程 / m",fontsize=7)
    panel(b,"b","待测点云的拟合保留掩膜")
    keep=data["moving_keep_mask"]
    b.scatter(mov[keep,0],mov[keep,1],s=.65,c=COLORS["pointwise_gate"],linewidths=0,rasterized=True,label="保留")
    b.scatter(mov[~keep,0],mov[~keep,1],s=1.2,c=COLORS["patch_candidate"],linewidths=0,rasterized=True,label="排除")
    spatial(b,bounds)
    b.legend(loc="upper center",bbox_to_anchor=(.5,-.25),ncol=2,markerscale=4,frameon=False,
             handletextpad=.4,borderaxespad=.1)
    panel(c,"c","输入坐标导出的残差筛选尺度")
    residual=np.r_[data["forward_centered_residual_m"][data["forward_valid"]],
                   data["reverse_centered_residual_m"][data["reverse_valid"]]]
    values=np.sort(abs(residual)*1000)
    c.plot(values,100*np.arange(1,len(values)+1)/len(values),lw=1,color=COLORS["pointwise_gate"])
    threshold=float(data["threshold_m"])*1000
    c.axvline(threshold,lw=.8,ls="--",color=COLORS["patch_candidate"])
    c.set(xlabel="双向有效对应 |r−μ| / mm",ylabel="累计比例 / %",xlim=(0,values.max()*1.04),ylim=(0,104))
    c.set_box_aspect(1)
    c.text(.41,.20,f"τ={threshold:.1f} mm\n有效对应n={len(values):,}",transform=c.transAxes,fontsize=7)
    panel(d,"d","筛选结果与已知变化的交叉计数")
    changed=data["moving_changed_evaluation_only"]
    counts=np.array([[np.sum(~changed & keep),np.sum(~changed & ~keep)],
                     [np.sum(changed & keep),np.sum(changed & ~keep)]])
    d.imshow(counts,cmap="Blues",norm=Normalize(0,max(counts.max(),1)),aspect="auto")
    for (row,col),count in np.ndenumerate(counts):
        d.text(col,row,f"{count:,}",ha="center",va="center",fontsize=9,
               color="white" if count>counts.max()*.6 else "#243447")
    d.set_xticks([0,1],["拟合保留","拟合排除"]);d.set_yticks([0,1],["实际未变","实际变化"])
    d.set_xlabel("算法筛选");d.set_ylabel("真值仅用于评价");d.set_box_aspect(1)
    for ax in [c,d]:ax.tick_params(length=2.5,width=.6)
    meta["Fig02"]={"reference_points":len(ref),"moving_points":len(mov),
        "retained_moving":int(keep.sum()),"excluded_moving":int((~keep).sum()),
        "cdf_valid_correspondences":len(values),"cdf_exclusion_rule":"3s Euclidean match gate, identical to algorithm",
        "confusion_matrix_rows_unchanged_changed_columns_keep_exclude":counts.tolist(),
        "changed_exclusion_recall":float(np.mean(~keep[changed]))}
    write_csv(output/"source_data/selection_confusion.csv",["true_class","kept","excluded"],
              [["unchanged",*counts[0]],["changed",*counts[1]]])
    export(fig,"Fig02_reference_selection",output,axes=list(axs.ravel()),exclude_axes=[ca])


def heights(output, grids, rows, meta):
    fig,axs=plt.subplots(2,3,figsize=(7.20472440945,5.94488188976))
    fig.subplots_adjust(left=.08,right=.86,bottom=.10,top=.82,wspace=.40,hspace=.60)
    heading(fig,"图3  配准后高差及位姿影响","上排：估计高差；下排：相对正确位姿对照的高差差异（非总真值误差）")
    methods=["ordinary","huber","pointwise_gate"]
    shape=tuple(grids["grid_shape"]); xy=grids["xy"]; step=float(grids["cell_size_m"])
    extent=(xy[:,0].min()-step/2,xy[:,0].max()+step/2,xy[:,1].min()-step/2,xy[:,1].max()+step/2)
    valid=grids["common_valid"]; truth=grids["true_delta_z_evaluation_only"]
    meta["Fig03"]={"common_cells":int(valid.sum()),"cell_size_m":step,"color_clipping":{},
        "bottom_panel_reference":"estimated z minus oracle-pose z; NOT total height error to truth"}
    for j,method in enumerate(methods):
        delta=grids[method+"_z"]-grids["reference_z"]
        error=(grids[method+"_z"]-grids["oracle_pose_z"])*1000
        record=next(r for r in rows if r["case"]==CASE and r["method"]==method)
        values=[delta,error]; norms=[TwoSlopeNorm(vmin=-.5,vcenter=0,vmax=.5),TwoSlopeNorm(vmin=-20,vcenter=0,vmax=20)]
        for i,(value,norm) in enumerate(zip(values,norms)):
            ax=axs[i,j]
            title=LABELS[method] if i==0 else f"体积误差 {float(record['net_absolute_error_m3']):.3f} m³"
            panel(ax,chr(97+i*3+j),title)
            masked=np.ma.array(value.reshape(shape),mask=~valid.reshape(shape))
            im=ax.imshow(masked,origin="lower",extent=extent,cmap="RdBu_r",norm=norm,interpolation="nearest",rasterized=True)
            spatial(ax,(-23,23,-23,23),xlabel=i==1,ylabel=j==0)
            if j>0:ax.set_yticklabels([])
            clipped=(value[valid]<norm.vmin)|(value[valid]>norm.vmax)
            meta["Fig03"]["color_clipping"][method+("_delta" if i==0 else "_error")]=int(clipped.sum())
        assert abs(delta[valid].sum()*step**2-float(record["estimated_common_net_m3"]))<1e-8
    fig.canvas.draw(); color_axes=[]
    for i in range(2):
        pos=axs[i,-1].get_position(); ca=fig.add_axes([.89,pos.y0,.015,pos.height]);color_axes.append(ca)
        cb=fig.colorbar(axs[i,-1].images[0],cax=ca,extend="both")
        cb.set_label("高差 / m" if i==0 else "位姿引入的高差差异 / mm",fontsize=7)
        cb.set_ticks([-.5,0,.5] if i==0 else [-20,0,20])
    export(fig,"Fig03_height_difference_and_bias",output,axes=list(axs.ravel()),exclude_axes=color_axes)


def overall(output, rows, meta):
    fig,axs=plt.subplots(1,2,figsize=(7.20472440945,4.09448818898))
    fig.subplots_adjust(left=.20,right=.96,bottom=.19,top=.79,wspace=.80)
    heading(fig,"图4  全部方法的总体比较与逐案例配对","21例＝7种条件×3个种子；所有条件均来自同一地形片区")
    a,b=axs;panel(a,"a","全部方法的体积误差")
    for i,method in enumerate(METHODS):
        records=sorted([r for r in rows if r["method"]==method],key=lambda r:r["case"])
        values=np.array([float(r["net_absolute_error_m3"]) for r in records])
        assert len(values)==21 and np.isfinite(values).all()
        jitter=np.linspace(-.20,.20,len(values))
        a.scatter(values,i+jitter,s=9,color=COLORS[method],alpha=.68,linewidths=0,zorder=3)
        a.errorbar(values.mean(),i,xerr=values.std(ddof=1),fmt="D",ms=3.5,lw=.85,
                   color=COLORS[method] if method=="pointwise_gate" else "#344457",capsize=2,zorder=4)
    a.set_yticks(range(len(METHODS)),[LABELS[m] for m in METHODS]);a.invert_yaxis()
    a.set_xlabel("绝对误差 / m³");a.grid(axis="x",color="#E5E9EE",lw=.5,zorder=0)
    a.set_xlim(left=min(-.25,a.get_xlim()[0]))
    panel(b,"b","主方案与Huber配对误差")
    records={r["case"]:{rr["method"]:rr for rr in rows if rr["case"]==r["case"]} for r in rows}
    maximum=0
    for case in sorted(records):
        one=records[case];x=float(one["huber"]["net_absolute_error_m3"]);y=float(one["pointwise_gate"]["net_absolute_error_m3"])
        sparse=one["huber"]["scenario"]=="deposition_sparse"
        b.scatter(x,y,s=18 if sparse else 12,marker="D" if sparse else "o",
                  color=COLORS["patch_candidate"] if sparse else COLORS["pointwise_gate"],lw=0,zorder=3)
        maximum=max(maximum,x,y)
    lim=maximum*1.1;b.plot([0,lim],[0,lim],color="#8B97A3",ls="--",lw=.7,zorder=1)
    b.set(xlim=(-.05,lim),ylim=(-.05,lim),xlabel="Huber体积绝对误差 / m³",ylabel="主方案体积绝对误差 / m³")
    fig.text(.20,.075,"a：点为单例；菱形与误差棒为均值±样本标准差。",fontsize=7,color="#546171")
    fig.text(.20,.035,"b：虚线为误差相等；橙色菱形为变稀条件，其余为蓝点。",fontsize=7,color="#546171")
    wins=sum(float(v["pointwise_gate"]["net_absolute_error_m3"])<float(v["huber"]["net_absolute_error_m3"]) for v in records.values())
    meta["Fig04"]={"cases":21,"method_count":8,"main_better_than_huber_cases":wins,
        "spread":"sample SD across heterogeneous cases; not a CI", "raw_points_per_method":21}
    export(fig,"Fig04_overall_and_paired_comparison",output,axes=list(axs))


def robustness(output, rows, meta):
    fig,axs=plt.subplots(2,1,figsize=(7.20472440945,5.86614173228))
    fig.subplots_adjust(left=.12,right=.97,bottom=.14,top=.79,hspace=.48)
    heading(fig,"图5  不同条件下的收益与适用边界","每种条件3个同种子配对案例；点为均值，误差棒为样本标准差")
    a,b=axs;panel(a,"a","体积误差的分条件比较")
    display=["huber","pointwise_gate","patch_candidate","oracle_pose"]
    summary=[]
    for j,method in enumerate(display):
        means=[];sds=[]
        for scenario in SCENARIOS:
            values=np.array([float(r["net_absolute_error_m3"]) for r in rows if r["method"]==method and r["scenario"]==scenario])
            assert len(values)==3
            means.append(values.mean());sds.append(values.std(ddof=1))
            summary.append([scenario,method,3,values.mean(),values.std(ddof=1)])
        a.errorbar(np.arange(7)+(j-1.5)*.14,means,yerr=sds,fmt=["s","o","^","D"][j],
                   ms=3.2,ls="none",color=COLORS[method],lw=.8,capsize=2,label=LABELS[method])
    a.set_ylabel("体积绝对误差 / m³");a.grid(axis="y",lw=.5,color="#E5E9EE")
    a.set_xticks(range(7),SCENARIO_LABELS);a.set_xlim(-.55,6.55)
    fig.legend(*a.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.56,.875),
               ncol=4,frameon=False,columnspacing=1.4,handletextpad=.4)
    panel(b,"b","配对收益：Huber误差减主方案误差")
    differences=[]
    for i,scenario in enumerate(SCENARIOS):
        one=[r for r in rows if r["scenario"]==scenario]
        paired=[]
        for seed in sorted({r["seed"] for r in one}):
            h=next(float(r["net_absolute_error_m3"]) for r in one if r["method"]=="huber" and r["seed"]==seed)
            p=next(float(r["net_absolute_error_m3"]) for r in one if r["method"]=="pointwise_gate" and r["seed"]==seed)
            paired.append(h-p);differences.append([scenario,seed,h,p,h-p])
        b.scatter(i+np.array([-.10,0,.10]),paired,s=12,color=COLORS["pointwise_gate"],zorder=3,lw=0)
        b.errorbar(i,np.mean(paired),yerr=np.std(paired,ddof=1),fmt="_",ms=10,color="#344457",lw=.8,capsize=2,zorder=4)
    b.axhline(0,color="#8B97A3",ls="--",lw=.7);b.set_ylabel("配对误差差值 / m³")
    b.set_xticks(range(7),SCENARIO_LABELS);b.set_xlim(-.55,6.55)
    fig.text(.12,.045,"b：正值表示主方案误差更低；负值表示Huber更低。全部21个配对值均显示。",fontsize=7,color="#546171")
    write_csv(output/"source_data/scenario_statistics.csv",["scenario","method","n","mean_m3","sample_sd_m3"],summary)
    write_csv(output/"source_data/paired_huber_minus_main.csv",["scenario","seed","huber_error_m3","main_error_m3","paired_difference_m3"],differences)
    meta["Fig05"]={"cases_per_condition":3,"conditions":7,"spread":"sample SD, ddof=1; paired differences computed within seed"}
    export(fig,"Fig05_robustness_and_boundaries",output,axes=list(axs))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output",type=Path,default=PROJECT/"reports/paper_figures_v1")
    ap.add_argument("--source-data",type=Path)
    args=ap.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    (output/"qa").mkdir(exist_ok=True)
    source=args.source_data.resolve() if args.source_data else output/"source_data"
    if not (source/"representative_points.npz").exists():prepare_sources(source)
    (output/"source_data").mkdir(exist_ok=True)
    rows_all=read_csv(source/"all_results.csv")
    rows=[r for r in rows_all if r["split"]=="new_conditions"]
    assert len(rows)==168 and len({r["case"] for r in rows})==21
    with np.load(source/"representative_points.npz") as z: data={k:z[k] for k in z.files}
    with np.load(source/"height_grids.npz") as z: grids={k:z[k] for k in z.files}
    meta={"backend":"Python/matplotlib","date":"2026-11-24","width_mm":183,"png_dpi":600,
          "rows_before":len(rows_all),"rows_displayed_in_main_comparison":len(rows),
          "exclusion":"16 rows belonging to 2 development cases excluded from the 21-case new-conditions comparison",
          "representative_case":CASE,"point_cloud_subsampling_for_display":False,
          "algorithm_parameters_changed":False,"palette":COLORS}
    for function,arguments in [(flow,(output,)),(selection,(output,data,meta)),
        (heights,(output,grids,rows_all,meta)),(overall,(output,rows,meta)),(robustness,(output,rows,meta))]:
        function(*arguments);print(function.__name__+": exported",flush=True)
    meta["source_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
    (output/"figure_manifest.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    source_copy=output/"source";source_copy.mkdir(exist_ok=True)
    if Path(__file__).resolve() != (source_copy/"plot_paper_figures.py").resolve():
        shutil.copy2(__file__,source_copy/"plot_paper_figures.py")
    helpers=source_copy/"figure_helpers";helpers.mkdir(exist_ok=True)
    helper_source=Path(__file__).parent/"figure_helpers/audit_panel_alignment.py"
    if helper_source.resolve() != (helpers/"audit_panel_alignment.py").resolve():
        shutil.copy2(helper_source,helpers/"audit_panel_alignment.py")


if __name__=="__main__":main()
