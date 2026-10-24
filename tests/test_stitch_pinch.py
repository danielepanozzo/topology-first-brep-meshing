"""stitch_untraced_boundary_loops: fill clean untraced holes; refuse a hole that
touches another boundary loop at one vertex (a pinch), where igl.all_boundary_loop
returns fragments whose steps are not mesh edges, so the caller falls back to
the robust tracer instead of building faces on edges that do not exist."""
import numpy as np
import igl

from operations.extract import stitch_untraced_boundary_loops
from utils.mesh_utils import create_edge

N = 8
HOLE = {(i, j) for i in range(1, 4) for j in range(1, 4)}  # untraced hole: cells [1, 4)^2


def vid(i, j):
    return i * (N + 1) + j


def grid(removed):
    """N x N grid of CCW triangles without the cells in `removed`."""
    faces = []
    for i in range(N):
        for j in range(N):
            if (i, j) not in removed:
                faces.append([vid(i, j), vid(i, j + 1), vid(i + 1, j + 1)])
                faces.append([vid(i, j), vid(i + 1, j + 1), vid(i + 1, j)])
    V = [[float(i), float(j), 0.0] for i in range(N + 1) for j in range(N + 1)]
    return V, faces


def boundary_half_edges(faces):
    he = {(f[k], f[(k + 1) % 3]) for f in faces for k in range(3)}
    return {(a, b) for (a, b) in he if (b, a) not in he}


def on_hole_square(a, b):
    """Edge (a, b) lies on the border of the hole square [1, 4]^2."""
    (ia, ja), (ib, jb) = divmod(a, N + 1), divmod(b, N + 1)
    inside = all(1 <= x <= 4 for x in (ia, ja, ib, jb))
    return inside and ((ia == ib and ia in (1, 4)) or (ja == jb and ja in (1, 4)))


def traced_except_hole(faces):
    return {create_edge(a, b) for a, b in boundary_half_edges(faces) if not on_hole_square(a, b)}


def test_clean_hole_is_filled():
    V, faces = grid(HOLE)
    traced = traced_except_hole(faces)
    bnd = boundary_half_edges(faces)
    n_V = len(V)
    new_faces, ok = stitch_untraced_boundary_loops(V, faces, traced)
    assert ok
    assert len(V) == n_V + 1  # one apex
    fan = new_faces[len(faces):]
    assert len(fan) == 12
    assert all((a, b) in bnd for b, a, _ in fan)  # each fan face sits on a boundary half-edge
    F = np.array(new_faces)
    assert igl.is_edge_manifold(F)
    assert [len(loop) for loop in igl.all_boundary_loop(F)] == [4 * N]


def test_second_traced_loop_is_kept():
    """Outer loop + traced inner hole + untraced hole: only the untraced hole is
    filled (20622 patch 188 has two traced loops)."""
    V, faces = grid(HOLE | {(i, j) for i in range(5, 7) for j in range(5, 7)})
    n_V = len(V)
    new_faces, ok = stitch_untraced_boundary_loops(V, faces, traced_except_hole(faces))
    assert ok
    assert len(V) == n_V + 1
    assert len(new_faces) - len(faces) == 12
    assert sorted(len(loop) for loop in igl.all_boundary_loop(np.array(new_faces))) == [8, 4 * N]


def test_partially_traced_hole_is_not_stitched():
    V, faces = grid(HOLE)
    traced = traced_except_hole(faces) | {create_edge(vid(1, 1), vid(1, 2))}
    n_V = len(V)
    new_faces, ok = stitch_untraced_boundary_loops(V, faces, traced)
    assert not ok
    assert len(V) == n_V
    assert new_faces == faces


def test_pinched_hole_is_not_stitched():
    """Removing cell (0, 0) makes the hole touch the outer loop at vid(1, 1)."""
    V, faces = grid(HOLE | {(0, 0)})
    bnd = boundary_half_edges(faces)
    loops = [[int(v) for v in loop] for loop in igl.all_boundary_loop(np.array(faces))]
    assert any((loop[i - 1], loop[i]) not in bnd for loop in loops for i in range(len(loop)))
    n_V = len(V)
    new_faces, ok = stitch_untraced_boundary_loops(V, faces, traced_except_hole(faces))
    assert not ok
    assert len(V) == n_V
    assert new_faces == faces
