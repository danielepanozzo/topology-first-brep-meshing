"""Regression tests for the meshing-group fixes (main.py, initial_meshing.py,
run_one_patch.py, utils/regular_space.py)."""
import pytest

from utils.data_structures import LoopTracingData, PatchRegularSpaceData, PatchTracingData


def test_corner_stitch_without_path_raises():
    # corner 5 has vid 0 in the initial loop and vid 3 in loop 1, on two separate
    # triangles: no path joins them, so the corner cannot be stitched
    from run_one_patch import edit_loops_with_shared_corners
    V = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]]
    F = [[0, 1, 2], [3, 4, 5]]
    ptd = PatchTracingData()
    ptd.initial_loop_tracing_data.loop_tracing_data.corner_id_to_vid = {5: 0}
    ptd.remaining_loops_tracing_data = {1: LoopTracingData(corner_id_to_vid={5: 3})}
    with pytest.raises(ValueError, match="No path from apex vid 0 to vid 3"):
        edit_loops_with_shared_corners(V, F, PatchRegularSpaceData(), ptd)
    assert ptd.remaining_loops_tracing_data[1].corner_id_to_vid == {5: 3}


def test_fallback_mesh_keeps_its_sub_domains(monkeypatch):
    # The per-sub-domain mmg call fails after decide_trim_domain returned new
    # sub-domains: the fallback is still the initial grid, so it must come with
    # the grid's sub-domains, both for the fallback processing and the next attempt.
    import numpy as np
    import initial_meshing as im
    grid_V = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    grid_F = np.array([[0, 1, 2], [0, 2, 3]])
    grid_sd = [(0., 1., 0., 1.)]
    new_sd = [(0., 0.4, 0., 1.), (0.6, 1., 0., 1.)]

    def create_initial_mesh(face, u_lower=None, **kw):
        return grid_V.copy(), grid_F.copy(), {}, False, grid_sd

    def remesh(surface, uvs, F, **kw):
        if remesh.calls:
            raise RuntimeError("mmg failed")
        remesh.calls += 1
        return uvs * 0.5, F, 1.0
    remesh.calls = 0

    seen = {}

    def simple_processing(V, F, face, sub_domains, *rest):
        seen["sub_domains"] = sub_domains
        return (None,) * 6

    class Surface:
        def sample(self, uv):
            return np.c_[uv, np.zeros(len(uv))]

    class Face:
        id, singularities, surface = 0, {}, Surface()

    monkeypatch.setattr(im, "curvature_capped_error", lambda surface, e: e)
    monkeypatch.setattr(im, "create_initial_mesh", create_initial_mesh)
    monkeypatch.setattr(im, "detect_periodicity", lambda *a: (False, False))
    monkeypatch.setattr(im, "remesh_with_error_and_edge_metric", remesh)
    monkeypatch.setattr(im, "detect_collapsed_iso_lines", lambda *a, **kw: {})
    monkeypatch.setattr(im, "decide_trim_domain", lambda face, **kw: (new_sd, 0.1, False, 0))
    monkeypatch.setattr(im, "insert_feature_edges_uv", lambda V, F, *a, **kw: (V, F))
    monkeypatch.setattr(im, "initial_mesh_simple_processing", simple_processing)
    with pytest.warns(UserWarning, match="mmg failed"):
        *_, V_fb, F_fb, sd_fb = im._run_initial_meshing_for_face(Face(), {}, target_error=0.1)
    assert np.array_equal(V_fb, grid_V) and np.array_equal(F_fb, grid_F)
    assert sd_fb == grid_sd
    assert seen["sub_domains"] == grid_sd


def test_init_grid_mesh_rejects_two_interior_singularities():
    from initial_meshing import init_grid_mesh

    def sing(u):
        return {'first2d': (u, 0.), 'uiso': True, 'point2d': (u, 0.5), 'point3d': (0., 0., 0.)}
    # one interior singularity is supported: two grids, no unfilled triangle
    uvs, F, _, sub_domains = init_grid_mesh({'000': sing(0.5)}, 4, 4, 0., 1., 0., 1.)
    assert len(sub_domains) == 2 and (F != 0).any(axis=1).all()
    with pytest.raises(ValueError, match="at most one interior singularity"):
        init_grid_mesh({'000': sing(0.3), '001': sing(0.7)}, 4, 4, 0., 1., 0., 1.)


def traced_ptd():
    """Corner 5 at vid 0 (initial loop) and vid 3 (loop 1); feature 1 traced as 0-1-2."""
    ptd = PatchTracingData()
    lt = ptd.initial_loop_tracing_data.loop_tracing_data
    lt.corner_id_to_vid = {5: 0}
    lt.feature_id_to_edge_tracing_data = {1: [0, 1, 2]}
    ptd.remaining_loops_tracing_data = {1: LoopTracingData(corner_id_to_vid={5: 3})}
    return ptd


def test_frozen_and_traced_edge_sets_are_separate_after_construction():
    from utils.regular_space import construct_patch_regular_space_from_tracing_data
    rsd = PatchRegularSpaceData()
    construct_patch_regular_space_from_tracing_data(rsd, traced_ptd())
    assert rsd.frozen_edges_set == rsd.traced_edge_set == {(0, 1), (1, 2)}
    rsd.frozen_edges_set.add((2, 3))
    assert (2, 3) not in rsd.traced_edge_set


def test_frozen_and_traced_edge_sets_are_separate_when_stitching(monkeypatch):
    import run_one_patch
    seen = []

    def stitch(V, connectivity, corner_id, vids, corners_to_process, rsd, ptd):
        seen.append(rsd.frozen_edges_set is not rsd.traced_edge_set)
        rsd.frozen_edges_set.add((2, 3))  # a stitch path is frozen, not traced
        assert (2, 3) not in rsd.traced_edge_set
        return None
    monkeypatch.setattr(run_one_patch, "trace_and_collapse_between_two_vertices", stitch)
    V = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    run_one_patch.edit_loops_with_shared_corners(V, [[0, 1, 2], [0, 2, 3]], PatchRegularSpaceData(), traced_ptd())
    assert seen == [True]
