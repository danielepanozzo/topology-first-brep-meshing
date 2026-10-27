"""Regression tests for the remesh_background / embed_geodesic fixes."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

import operations.embed_geodesic as eg
from operations.remesh_background import refine_cdt_with_size_field, write_medit_sol


class Plane:
    """z = 0 plane parameterized by (u, v)."""

    def sample(self, uv):
        uv = np.asarray(uv, dtype=float)
        return np.column_stack([uv, np.zeros(len(uv))])


# Unit square, two triangles, diagonal (0, 2) as the only split segment.
SQ_V = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SQ_F = np.array([[0, 1, 2], [0, 2, 3]])


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


def _raise(exc):
    def f(*args, **kwargs):
        raise exc
    return f


def test_cdt_rebuild_lets_timeout_through(monkeypatch):
    """alarm_timeout's TimeoutError during the plain CDT rebuild reaches the
    caller instead of falling back to the fan-split mesh."""
    import triangle
    monkeypatch.setattr(triangle, "triangulate", _raise(TimeoutError("alarm")))
    segs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    marks = [30, 30, 30, 30, 1]
    size_field = (cKDTree(Plane().sample(SQ_V)), np.full(4, 0.25))
    with pytest.raises(TimeoutError):
        refine_cdt_with_size_field(SQ_V, SQ_F, segs, marks, {1}, Plane(), size_field)


def _embed(monkeypatch, **patches):
    for name, fn in patches.items():
        monkeypatch.setattr(eg, name, fn)
    face = SimpleNamespace(loops=[], surface=Plane(), singularities=None)
    return eg.embed_loops_one_patch(
        SQ_V, Plane().sample(SQ_V), SQ_F, SQ_F, face, {}, [(0.0, 1.0, 0.0, 1.0)],
        before_embedding_edge_length=0.25, mmg_factor=1.0)


def test_conflict_refine_lets_timeout_through(monkeypatch):
    with pytest.raises(TimeoutError):
        _embed(monkeypatch,
               refine_background_for_feature_conflicts=_raise(TimeoutError("alarm")),
               embed_geodesic_segments_in_uv=_raise(AssertionError("timeout swallowed")))


def test_cdt_refine_pass_lets_timeout_through(monkeypatch):
    with pytest.raises(TimeoutError):
        _embed(monkeypatch,
               refine_background_for_feature_conflicts=lambda V, F, *a, **k: (V, F),
               embed_geodesic_segments_in_uv=lambda *a, **k: (
                   SQ_V, SQ_F, np.array([[0, 2]]), {}),
               refine_cdt_with_size_field=_raise(TimeoutError("alarm")),
               compute_first_fundamental_form=_raise(AssertionError("timeout swallowed")))
