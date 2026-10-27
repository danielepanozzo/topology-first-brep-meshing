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
