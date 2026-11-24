"""Compare py4dgeo native transformation with its documented NumPy equation."""
from pathlib import Path
import json
import numpy as np
import py4dgeo as pg
import _py4dgeo
from py4dgeo.registration import _fit_transform

root = Path(__file__).resolve().parents[1]
a, b = [np.loadtxt(root / 'usage_data' / f'plane_horizontal_t{i}.xyz') for i in (1, 2)]
trafo = _fit_transform(b, a)
expected = b @ trafo[:3, :3].T + trafo[:3, 3]
epoch = pg.Epoch(b.copy())
epoch.transform(affine_transformation=trafo, reduction_point=np.zeros(3))
delta = np.asarray(epoch.cloud) - expected
result = {'native_vs_numpy_max_abs_m': float(np.max(abs(delta))),
          'native_transform_matches_formula': bool(np.allclose(epoch.cloud, expected, atol=1e-12)),
          'first_five_native_errors_m': delta[:5].tolist()}
cloud = pg.Epoch(b.copy()).cloud.copy()
_py4dgeo.transform_pointcloud_inplace(cloud, trafo, np.zeros(3), np.empty((1, 3)))
result['raw_native_vs_numpy_max_abs_m'] = float(np.max(abs(cloud - expected)))
result['raw_native_input_flags'] = str(cloud.flags)
u, singular_values, vt = np.linalg.svd((b-b.mean(0)).T @ (a-a.mean(0)))
correct_translation = a.mean(0) - trafo[:3, :3] @ b.mean(0)
result['reflection_branch'] = {
    'determinant_before_correction': float(np.linalg.det(vt.T @ u.T)),
    'returned_translation':trafo[:3, 3].tolist(),
    'translation_required_by_returned_rotation':correct_translation.tolist(),
    'translation_error_m':(trafo[:3, 3]-correct_translation).tolist(),
    'singular_values':singular_values.tolist()}
for i in (1, 2):
    e = pg.read_from_las(str(root/'usage_data'/f'plane_horizontal_t{i}_w_normals.laz'), normal_columns=['NormalX','NormalY','NormalZ'])
    original = a if i == 1 else b
    result[f'provided_t{i}'] = {'coordinate_max_difference':float(np.max(abs(e.cloud-original))),
                               'normals_min':e.normals.min(0).tolist(), 'normals_max':e.normals.max(0).tolist()}
print(json.dumps(result, indent=2))
(root / 'logs' / 'native_transform_diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
