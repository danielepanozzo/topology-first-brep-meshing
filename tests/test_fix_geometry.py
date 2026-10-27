"""Regression tests for fixes in the geometry code (alignment, snapping, mesh utilities,
consolidation), on small hand-built inputs."""
import numpy as np
import pytest

from operations.align import align_one_segment_of_patches
from utils.data_structures import ConnectivityMaintainance
from utils.mesh_utils import adjacency_list, compute_VF, edge_length_mapping


def fan_patch(xs, apex):
    """Triangle fan over the polyline (x, 0, 0), x in xs, to one apex vertex."""
    V = [np.array([x, 0.0, 0.0]) for x in xs] + [np.array(apex, dtype=float)]
    a = len(xs)
    F = [[k, k + 1, a] for k in range(len(xs) - 1)]
    conn = ConnectivityMaintainance(F, compute_VF(F), adjacency_list(F, len(V)),
                                    edge_length_mapping(V, F), {}, set())
    return V, conn


@pytest.mark.parametrize("scale", [0.1, 100.0])
def test_align_snaps_by_normalized_parameter_at_any_scale(scale):
    # Interior vertices at normalized parameters 0.2, 0.6 (patch A) and 0.4, 0.605 (patch B):
    # only 0.6 and 0.605 are within 1% of the segment, at any model scale.
    VA, cA = fan_patch([0.0, 0.2 * scale, 0.6 * scale, scale], [0.5 * scale, scale, 0.0])
    VB, cB = fan_patch([0.0, 0.4 * scale, 0.605 * scale, scale], [0.5 * scale, -scale, 0.0])
    Vs = [VA, VB]
    segments = [[0, 1, 2, 3], [0, 1, 2, 3]]
    aligned = align_one_segment_of_patches(Vs, [cA, cB], segments)
    expected = [0.0, 0.2, 0.4, 0.6025, 1.0]
    for V, conn, vids in zip(Vs, [cA, cB], aligned):
        chain = vids + [3]  # the returned chain leaves out the segment's last vertex
        assert all(b in conn.adj_list[a] for a, b in zip(chain, chain[1:]))
        assert np.allclose([V[v][0] / scale for v in chain], expected)
