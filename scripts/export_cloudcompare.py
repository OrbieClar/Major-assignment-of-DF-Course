"""Export true-coordinate colored PLY overlays for CloudCompare.

No registration is recomputed. Scalar fields preserve epoch ID and residuals.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cloudcompare"
OUT.mkdir(parents=True, exist_ok=True)
BLUE = (40, 135, 235)
ORANGE = (245, 130, 35)


def nearest_mm(cloud, reference):
    return np.linalg.norm(cloud[:, None, :] - reference[None, :, :], axis=2).min(axis=1) * 1000


def write_ply(path, xyz, colors, epoch_ids, distances):
    header = [
        "ply", "format ascii 1.0",
        "comment Coordinates are unchanged from original/saved registration outputs",
        "comment Blue = reference t1; orange = moving t2",
        "comment scalar_nn_distance_mm = distance to nearest reference t1 point",
        f"element vertex {len(xyz)}",
        "property double x", "property double y", "property double z",
        "property uchar red", "property uchar green", "property uchar blue",
        "property float scalar_epoch_id", "property double scalar_nn_distance_mm",
        "end_header",
    ]
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("\n".join(header) + "\n")
        for point, color, epoch, distance in zip(xyz, colors, epoch_ids, distances):
            file.write("{:.12f} {:.12f} {:.12f} {} {} {} {} {:.12f}\n".format(*point, *color, int(epoch), distance))
    # Validate the actual saved file, including RGB and custom scalar properties.
    lines = path.read_text(encoding="ascii").splitlines()
    saved = np.loadtxt(lines[lines.index("end_header") + 1:])
    assert saved.shape == (len(xyz), 8), path
    assert np.allclose(saved[:, :3], xyz, rtol=0, atol=1e-11), path
    assert np.array_equal(saved[:, 3:6], colors), path
    assert np.array_equal(saved[:, 6], epoch_ids), path
    assert np.allclose(saved[:, 7], distances, rtol=0, atol=1e-10), path


reference_path = ROOT / "usage_data" / "plane_horizontal_t1.xyz"
reference = np.loadtxt(reference_path)[:, :3]
blue = np.tile(BLUE, (len(reference), 1))
write_ply(OUT / "reference_t1_blue.ply", reference, blue, np.ones(len(reference)), np.zeros(len(reference)))

stages = [
    ("01_before", ROOT / "usage_data" / "plane_horizontal_t2.xyz", "配准前"),
    ("02_after_ordinary", ROOT / "results" / "numpy_baseline_venv" / "ordinary_aligned.xyz", "普通 ICP 后（NumPy，Open3D 核对）"),
    ("03_after_py4dgeo_ordinary", ROOT / "results" / "py4dgeo_comparison" / "ordinary_aligned.xyz", "py4dgeo 原始普通 ICP 后"),
    ("04_after_stable", ROOT / "results" / "py4dgeo_comparison" / "stable_aligned.xyz", "py4dgeo 稳定区域 ICP 后（本次误差增大）"),
]
manifest = {"format": "ASCII PLY 1.0", "coordinates": "unchanged; no offsets or exaggeration",
            "coordinate_unit": "m (same assumption as registration report)",
            "reference_rgb": BLUE, "moving_rgb": ORANGE, "outputs": []}
table = []
for prefix, source_path, label in stages:
    moving = np.loadtxt(source_path)[:, :3]
    distances = nearest_mm(moving, reference)
    orange = np.tile(ORANGE, (len(moving), 1))
    single = f"{prefix}_t2_orange.ply"
    overlay = f"{prefix}_overlay.ply"
    write_ply(OUT / single, moving, orange, np.full(len(moving), 2), distances)
    write_ply(OUT / overlay, np.vstack((reference, moving)), np.vstack((blue, orange)),
              np.r_[np.ones(len(reference)), np.full(len(moving), 2)], np.r_[np.zeros(len(reference)), distances])
    rmse = float(np.sqrt(np.mean(distances**2)))
    manifest["outputs"].append({"stage": prefix, "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "moving_file": single, "overlay_file": overlay, "reference_points": len(reference),
        "moving_points": len(moving), "overlay_points": len(reference) + len(moving), "rmse_mm": rmse})
    table.append(f"| `{overlay}` | {label} | {rmse:.4f} mm |")
manifest["reference"] = {"source":str(reference_path), "sha256":hashlib.sha256(reference_path.read_bytes()).hexdigest()}
(OUT / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
readme = """# CloudCompare 点云对比文件

所有 PLY 使用真实坐标，未偏移、未放大 Z 轴。每个 overlay 文件包含两期点云共 882 点：**蓝色为参考时相 t1，橙色为待配准/配准后的 t2**。

## 建议先看

分别打开 `01_before_overlay.ply` 与 `02_after_ordinary_overlay.ply`，比较普通 ICP 配准前后。一次打开一个 overlay，避免多个阶段在相同位置互相遮挡。

| 文件 | 内容 | 全点最近邻 RMSE |
| --- | --- | ---: |
""" + "\n".join(table) + """

## 在 CloudCompare 中查看

1. 将所选 `.ply` 拖入 CloudCompare；导入对话框如出现，坐标使用 x/y/z，颜色使用 red/green/blue。
2. 采用 RGB 颜色显示。若正在显示标量场，关闭标量场显示或切回 RGB，恢复蓝/橙两色。
3. 点云为约 20×20 米的稀疏平面，配准前主要偏差只有约 0.1 米；俯视时不明显，建议旋转到侧视/低仰角并增大点尺寸查看。没有人为夸大坐标。
4. 若要分别隐藏或选择两期点云，同时导入 `reference_t1_blue.ply` 与某个 `*_t2_orange.ply`。两者在对象树中是独立点云，便于切换显示。overlay 是方便一次打开的合并版本。

额外标量字段：`epoch_id`（原始 PLY 属性为 `scalar_epoch_id`，1=t1，2=t2），`nn_distance_mm`（到参考时相 t1 最近点的距离，毫米）。参考点的此距离为 0；需查看 t2 残差分布时，打开独立的橙色 t2 文件并显示该标量场，避免混入参考点的零值。不同阶段采用不同自动色标范围时，不能只凭颜色比较误差大小。

稳定区域 ICP 本次误差增大，`04_after_stable_*` 保留的是实际输出，不代表配准成功。原始 py4dgeo 普通 ICP 的实现问题及详细运行结论见上级项目中的 `运行报告.md`。

导出脚本：`scripts/export_cloudcompare.py`；在项目根目录执行：

```powershell
& .\\.venv\\Scripts\\python.exe scripts\\export_cloudcompare.py
```

导出后已重新读取每个 PLY，核对点数、坐标、RGB、时相编号和逐点残差；来源及 SHA256 记录在 `export_manifest.json`。
"""
(OUT / "查看说明.md").write_text(readme, encoding="utf-8")
print(f"Exported and verified {1 + 2 * len(stages)} PLY files in {OUT}")
for entry in manifest["outputs"]:
    print(f"{entry['overlay_file']}: {entry['overlay_points']} points, RMSE {entry['rmse_mm']:.6f} mm")
