"""Uniform 2.5D TIN-height interpolation and signed grid-volume integration.

Cells outside the convex hull or supported only by long triangle edges / distant
points remain invalid. No filling with zero, min-height or a known truth surface.
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree


def interpolate_height(points, xy, maximum_distance, maximum_edge):
    triangle = Delaunay(points[:, :2])
    z = np.asarray(LinearNDInterpolator(triangle, points[:, 2])(xy))
    simplex = triangle.find_simplex(xy)
    vertices = points[triangle.simplices, :2]
    edge_max = np.maximum.reduce([np.linalg.norm(vertices[:, i]-vertices[:, j], axis=1)
                                  for i,j in ((0,1),(1,2),(2,0))])
    distance = cKDTree(points[:, :2]).query(xy)[0]
    valid = (simplex >= 0) & np.isfinite(z) & (distance <= maximum_distance)
    valid &= edge_max[np.maximum(simplex, 0)] <= maximum_edge
    z[~valid] = np.nan
    return z, valid


def integrate(dz, mask, step):
    selected = np.asarray(dz)[mask]
    if not np.isfinite(selected).all():
        raise ValueError("Invalid cells cannot enter volume integration")
    area = step**2
    return {"net_m3":float(selected.sum()*area),
            "added_m3":float(np.maximum(selected,0).sum()*area),
            "removed_m3":float(-np.minimum(selected,0).sum()*area),
            "covered_area_m2":float(mask.sum()*area)}


def analytical_checks():
    # Independent planar geometry: linear TIN interpolation must preserve a
    # constant vertical offset, with the correct sign and square-cell area.
    x,y = np.meshgrid(np.arange(0,4.01,.5),np.arange(0,4.01,.5))
    points = np.column_stack((x.ravel(),y.ravel(),(.2*x-.1*y).ravel()))
    gx,gy = np.meshgrid(np.arange(.125,4,.25),np.arange(.125,4,.25))
    query = np.column_stack((gx.ravel(),gy.ravel()))
    z0,valid0 = interpolate_height(points,query,.75,2.)
    raised = points.copy();raised[:,2] += .4
    z1,valid1 = interpolate_height(raised,query,.75,2.)
    common = valid0 & valid1
    positive = integrate(z1-z0,common,.25)
    negative = integrate(z0-z1,common,.25)
    expected = float(common.sum()*.25**2*.4)
    return {"constant_offset_volume":bool(abs(positive["net_m3"]-expected)<1e-10),
            "negative_volume_sign":bool(abs(negative["net_m3"]+expected)<1e-10),
            "added_removed_signs":bool(abs(positive["added_m3"]-expected)<1e-10 and abs(negative["removed_m3"]-expected)<1e-10),
            "expected_test_volume_m3":expected}
