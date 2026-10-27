"""Regression tests for the remesh_background / embed_geodesic fixes."""
import numpy as np
import pytest

from operations.remesh_background import write_medit_sol


def read_sol(path):
    """(nv, 3) tensors (m11, m12, m22) written by write_medit_sol."""
    lines = open(path).read().split("\n")
    start = lines.index("1 3") + 1
    nv = int(lines[start - 2])
    return np.array([[float(x) for x in lines[start + i].split()] for i in range(nv)])


def test_metric_clamp_is_scale_free(tmp_path):
    """A uniform scaling of the input (here 1/h^2 on a unit-speed plane, as the
    per-vertex size callers pass it) scales the written tensor by the same
    amount: no unit-dependent clamp to the identity at h = 2e-4 or h = 200."""
    path = str(tmp_path / "m.sol")
    # unit-speed plane, a skewed valid tensor and a pole (G = 0)
    E = np.array([1.0, 3.0, 1.0])
    F = np.array([0.0, 1.0, 0.0])
    G = np.array([1.0, 2.0, 0.0])
    write_medit_sol(path, E, F, G)
    ref = read_sol(path)
    for h in (2e-4, 200.0):
        k = 1.0 / h ** 2
        write_medit_sol(path, k * E, k * F, k * G)
        np.testing.assert_allclose(read_sol(path), k * ref, rtol=1e-14)
        np.testing.assert_allclose(read_sol(path)[0], [k, 0.0, k], rtol=1e-14)
        write_medit_sol(path, E, F, G, factor=k)
        np.testing.assert_allclose(read_sol(path), k * ref, rtol=1e-14)


def test_degenerate_metric_becomes_isotropic(tmp_path):
    path = str(tmp_path / "m.sol")
    E = np.array([2.0, 3.0, 9.0,     # valid, scales s = 2, 3, 5
                  1.0,               # pole: G = 0
                  1.0,               # not positive definite (det = -3)
                  4.0,               # anisotropic beyond 1e8:1
                  np.nan, -1.0])     # unusable scale -> median valid scale
    F = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    G = np.array([2.0, 3.0, 1.0, 0.0, 1.0, 1e-12, 1.0, -1.0])
    write_medit_sol(path, E, F, G, factor=10.0)
    expected = 10.0 * np.array([[2.0, 0.0, 2.0], [3.0, 0.0, 3.0], [9.0, 0.0, 1.0],
                                [0.5, 0.0, 0.5], [1.0, 0.0, 1.0],
                                [2.0 + 5e-13, 0.0, 2.0 + 5e-13],
                                [3.0, 0.0, 3.0], [3.0, 0.0, 3.0]])
    np.testing.assert_allclose(read_sol(path), expected, rtol=1e-14)


def test_metric_without_valid_scale_raises(tmp_path):
    with pytest.raises(ValueError):
        write_medit_sol(str(tmp_path / "m.sol"),
                        np.array([np.nan, 0.0]), np.zeros(2), np.array([1.0, 0.0]))
