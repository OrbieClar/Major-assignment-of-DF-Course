"""Summarize actual saved outputs, without substituting missing results."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
baseline = json.loads((root/'results/numpy_baseline_venv/metrics.json').read_text(encoding='utf-8'))
comparison = json.loads((root/'results/py4dgeo_comparison/metrics.json').read_text(encoding='utf-8'))
rows = [
    ('Before', baseline['before']['moving_to_reference_rmse_mm']),
    ('Ordinary ICP (NumPy; Open3D checked)', baseline['ordinary']['moving_to_reference_rmse_mm']),
    ('Ordinary ICP (py4dgeo 1.2.0 original)', comparison['ordinary']['moving_to_reference_rmse_mm']),
    ('Stable-area ICP (py4dgeo 1.2.0 original)', comparison['stable']['moving_to_reference_rmse_mm'])]
with (root/'results/comparison_summary.csv').open('w',newline='',encoding='utf-8-sig') as file:
    writer=csv.writer(file); writer.writerow(['method','all_point_nearest_neighbor_rmse_mm']); writer.writerows(rows)
fig, ax = plt.subplots(figsize=(10, 4.8))
bars = ax.barh([row[0] for row in rows], [row[1] for row in rows], color=['#94a3b8','#2588b8','#7180bd','#d7873c'])
ax.invert_yaxis(); ax.set_xlabel('All-point nearest-neighbor RMSE (mm)')
ax.set_xlim(0, max(row[1] for row in rows)*1.23)
for bar, (_, value) in zip(bars,rows):
    ax.text(value+2,bar.get_y()+bar.get_height()/2,f'{value:.4f}',va='center')
ax.set_title('Same 441-point epochs; t2 to t1; no residual trimming')
ax.spines[['top','right']].set_visible(False); fig.tight_layout()
fig.savefig(root/'results/comparison_summary.png', dpi=180)
plt.close(fig)
baseline_rmse, raw_rmse, stable_rmse = [row[1] for row in rows[1:]]
report=f'''# ICP 配准运行报告

## 执行结果

已在“大作业/.venv”建立独立 Python 3.13.15 环境，安装 py4dgeo 1.2.0。`pip check` 通过，完整实际依赖版本见 `requirements-lock.txt`。两种 ICP 已执行，结果均保存；“程序完成”不代表配准质量合格。

| 方法 | 全点最近邻 RMSE（mm） | 解释 |
| --- | ---: | --- |
| 配准前 | {rows[0][1]:.9f} | 相同的两期 XYZ 输入 |
| 普通点到点 ICP，NumPy 实现 | {baseline_rmse:.9f} | 独立环境重新运行；与已有 Open3D 结果一致 |
| py4dgeo 1.2.0 原始普通 ICP | {raw_rmse:.9f} | 保留库的原始行为，未修改安装包 |
| py4dgeo 1.2.0 稳定区域 ICP | {stable_rmse:.9f} | 本次配准变差，不能作为有效改进结果 |

![全点最近邻误差比较](results/comparison_summary.png)

## 输入与方法

`usage_data/plane_horizontal_t1.xyz` 为参考，`plane_horizontal_t2.xyz` 为待配准点云，各 441 点，方向为 t2→t1。沿用此前实验将坐标单位解释为米的约定。每种算法独立从原始 t2 开始，不串联配准。

稳定区域方法使用对应 `*_w_normals.laz` 的原有法向量。已检查 XYZ 点顺序与坐标匹配，坐标差低于 1e-8。参数：initial_distance_threshold=0.2，level_of_detection=0.015，reference_supervoxel_resolution=supervoxel_resolution=4.0，min_svp_num=10，reduction_point=[0,0,0]。使用 4 个 OpenMP 线程。稳定区域函数外围设有 15 秒 Python 迭代时间保护；主结果正常返回，没有触发超时。

评价对全部点计算欧氏最近邻距离，不截断、不筛掉大残差；同样保留反向和对称 RMSE、中位数及 95% 分位。没有变换真值，也没有独立稳定区域真值。全点评价中 fitness=1 按定义成立，不能作为成功证据。

## 发现的问题

py4dgeo 1.2.0 的 `_fit_transform` 在修正 SVD 的反射分支前计算平移量，修正旋转后未重新计算平移。本次输入确实触发该分支：原旋转行列式为 -1，初次拟合返回的 z 平移约 6.9002，而由修正后的旋转按质心公式计算应约 0.10015，差约 6.80006 个坐标单位。证据保存在 `logs/native_transform_diagnostic.json`；复现脚本为 `scripts/diagnose_native.py`。这是原始普通 ICP 与正确 Kabsch/NumPy/Open3D 输出不一致的已验证原因。没有改写第三方安装文件。

稳定区域 ICP 本次整体误差增加。数据基本为单一平面，平面内平移与绕法向旋转的约束较弱；这为点到面步骤的不稳定提供了合理解释，但没有通过对照试验确认唯一原因，因此不把它当作已证明的软件缺陷。不能用这组结果声称稳定区域方法优于普通 ICP。

另外保留了官方教程参数（阈值 10/10、超体素分辨率 0.2、最小点数 1）的流程试跑：`results/py4dgeo_provided_tutorial/` 使用自带法向，`results/py4dgeo_tutorial/` 使用估计法向。这些是流程检查，不应与主实验的检测阈值混淆。早先估计法向版本持续迭代后人工中止的日志为 `logs/estimated_normals_interrupted.log`；没有将其中止状态冒充完成结果。

## 输出与复现

- `results/numpy_baseline/`：系统 Python 初次复现结果及 Open3D 交叉验证。
- `results/numpy_baseline_venv/`：独立环境内的普通 ICP 结果。
- `results/py4dgeo_comparison/`：原始 py4dgeo 普通/稳定区域 ICP 的完整输出。
- 各结果目录中有 `manifest.json`、`metrics.csv/json`、配准点云 `*_aligned.xyz`、矩阵 `*_transformation.csv/json/npz`、逐点残差 `*_residuals.csv`、状态文件和验证记录。
- `logs/` 保存安装失败记录、安装后依赖检查、库原生日志和控制台日志。`requirements-lock.txt` 锁定实际安装版本。

在本目录执行：

```powershell
$env:OMP_NUM_THREADS = '4'
& .\\.venv\\Scripts\\python.exe scripts\\run_registration.py --engine numpy --method ordinary --output numpy_baseline_venv
& .\\.venv\\Scripts\\python.exe scripts\\run_registration.py --engine py4dgeo --method both
& .\\.venv\\Scripts\\python.exe scripts\\verify_results.py numpy_baseline_venv
& .\\.venv\\Scripts\\python.exe scripts\\verify_results.py py4dgeo_comparison
& .\\.venv\\Scripts\\python.exe scripts\\make_report.py
```

下一步适合用具有多方向几何约束的数据，并添加已知局部形变与已知刚体误差，检验稳定区选择的实际作用。本次已完成环境与两种算法运行，不将上述后续试验混入当前示例。

分享聊天链接未能读取；本次实现根据你的当前任务说明、本地数据和官方 API。原始数据只读，输入哈希核验通过。
'''
(root/'运行报告.md').write_text(report,encoding='utf-8')
print('Saved 运行报告.md and results/comparison_summary.csv/png')
