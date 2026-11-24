"""Package the finished report and reproducible experiment evidence."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/course_report_v1'

def ply(path,points,colors,extra=None):
    dtype=[('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1')]
    if extra is not None:dtype.append(('estimated_delta_z_m','<f8'))
    data=np.empty(len(points),dtype=dtype)
    for i,key in enumerate(('x','y','z')):data[key]=points[:,i]
    for i,key in enumerate(('red','green','blue')):data[key]=colors[:,i]
    if extra is not None:data['estimated_delta_z_m']=extra
    header='ply\nformat binary_little_endian 1.0\ncomment Local coordinates in metres; same origin as report\nelement vertex '+str(len(points))+'\n'
    for name,kind in dtype:header+='property '+('uchar' if kind=='u1' else 'double')+' '+name+'\n'
    header+='end_header\n'
    path.write_bytes(header.encode('ascii')+data.tobytes())

def representative_exports():
    folder=REPORT/'CloudCompare代表案例';folder.mkdir(exist_ok=True)
    with np.load(ROOT/'reports/paper_figures_v1/source_data/representative_points.npz') as z:
        ref=z['reference'];before=z['moving'];aligned=z['aligned_moving'];keep=z['moving_keep_mask']
    blue=np.tile([35,121,184],(len(ref),1));orange=np.tile([201,136,69],(len(aligned),1))
    ply(folder/'01_before_overlay.ply',np.vstack([ref,before]),np.vstack([blue,orange]))
    ply(folder/'02_pointwise_after_overlay.ply',np.vstack([ref,aligned]),np.vstack([blue,orange]))
    colors=np.tile([35,121,184],(len(aligned),1));colors[~keep]=[201,136,69]
    ply(folder/'03_pointwise_fit_selection.ply',aligned,colors)
    with np.load(ROOT/'reports/paper_figures_v1/source_data/height_grids.npz') as z:
        valid=z['common_valid'];xy=z['xy'][valid];z0=z['reference_z'][valid];z1=z['pointwise_gate_z'][valid]
    p=np.column_stack([xy,(z0+z1)/2]);ply(folder/'04_pointwise_height_difference.ply',p,np.tile([180,180,180],(len(p),1)),z1-z0)
    (folder/'查看说明.md').write_text('''# CloudCompare代表案例

标准堆积条件的首个预设种子，非按效果优选。

- 01：配准前叠加，参考蓝、待测橙。
- 02：主方案配准后完整点云叠加，参考蓝、待测橙。
- 03：主方案拟合保留掩膜；蓝色保留、橙色排除，仍显示全部待测点。
- 04：公共有效网格，高差在estimated_delta_z_m标量场中；Z坐标是两期平均高程，不是高差本身。

均为米制局部坐标，与报告一致。不再对每个文件分别平移或居中。生成真值只用于评价，未输入配准筛选。
''',encoding='utf-8')

def main():
    representative_exports()
    paths=set()
    def add(p):
        if p.is_file():paths.add(p)
    for root in ['scripts','configs']:
        for p in (ROOT/root).rglob('*'):
            if '__pycache__' not in p.parts and p.suffix in ['.py','.json','.md','.txt','.ps1']:add(p)
    for name in ['ahk2017_spatial_020m.ply','requirements.txt','requirements-lock.txt','README.md']:add(ROOT/name)
    for directory in ['terrain_benchmark_v1','stable_region_diagnosis_v1','automatic_stable_v1','robustness_volume_v1']:
        for p in (ROOT/'results'/directory).rglob('*'):
            if p.suffix in ['.npz','.json','.csv','.md']:add(p)
    for p in (ROOT/'logs').glob('*'):
        if p.suffix=='.log' and any(x in p.name for x in ['terrain_benchmark','stable_region_diagnosis','automatic_stable','robustness_volume','course_report_build']):add(p)
    for p in REPORT.rglob('*'):
        if 'qa_pages' in p.parts:continue
        if 'work' in p.parts and p.suffix not in ['.json','.txt']:continue
        if 'work' in p.parts and p.name=='report_page_plan.json':continue
        if p.suffix in ['.docx','.pdf','.md','.json','.txt','.ply','.png']:add(p)
    add(REPORT/'work/template_wordmark.png')
    for name in ['主方案说明_v1.md','报告大纲_v1.md']:add(ROOT/'reports'/name)
    paper=ROOT/'reports/paper_figures_v1'
    for p in paper.rglob('*'):
        if 'source' in p.parts:continue
        if p.suffix in ['.png','.svg','.pdf','.md','.json','.csv','.npz','.txt','.log']:
            if 'qa' in p.parts and p.suffix in ['.png','.svg','.pdf']:continue
            add(p)
    page_count=json.loads((REPORT/'document_build_manifest.json').read_text(encoding='utf-8'))['pdf_pages']
    readme=f'''# 大作业提交材料

主报告位于 大作业/reports/course_report_v1/，提供完整DOCX及{page_count}页PDF。
请先补填封面姓名、学号，并在Word打开后检查页码。评分页保持空白。

报告采用课程模拟专利结构，附五幅图、可编辑公式及真实实验结果；不将半合成数据宣称为真实两期变化真值。

代码、冻结配置、16,281点母本、两组开发数据和21例扩展输入/真值、算法矩阵与体积网格一并保存。
完整源工程的虚拟环境和原始大型下载包未打包；本报告所用的数据已保留。请在大作业目录运行脚本。

算法入口：scripts/run_robustness_volume.py
绘图入口：scripts/plot_paper_figures.py
报告入口：scripts/build_course_report.py（重建报告需Windows中文字体及NumPy/Matplotlib）
依赖版本：requirements-lock.txt；独立环境可另行按requirements.txt准备。

CloudCompare代表案例位于报告目录的CloudCompare代表案例文件夹。
报告中的限制、Word原生重分页未验证及图形碰撞审计未完成项，详见交付说明。

此包仅为整理好的本地材料，未发送邮件或执行对外提交。
'''
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    package=ROOT/'reports/大作业提交材料_v1.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('提交材料说明.md',readme)
        z.writestr('材料SHA256.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        for p in sorted(paths):z.write(p,'大作业/'+str(p.relative_to(ROOT)).replace('\\','/'))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert not any('/.venv/' in x or '__pycache__' in x for x in names)
        assert '大作业/reports/course_report_v1/2026秋-课程报告-说明书_完成稿.docx' in names
    print(json.dumps({'package':str(package),'files':len(paths)+2,'size_MB':round(package.stat().st_size/1024**2,2),'zip_test':'passed'},ensure_ascii=False))

if __name__=='__main__':main()
