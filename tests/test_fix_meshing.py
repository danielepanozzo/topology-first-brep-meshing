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
