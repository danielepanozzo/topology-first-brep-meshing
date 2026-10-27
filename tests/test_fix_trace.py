"""Regression tests for the fixes in operations/trace.py (initial and non-initial loop tracing)."""
import numpy as np
import pytest

import operations.trace as trace
from run_one_patch import construct_connectivity_data
from utils.data_structures import (HalfEdge, InitialLoopTracingData, LoopGeometryData, LoopTopologyData,
                                   LoopTracingData, NonInitialLoopTreatmentData, PatchRegularSpaceData)


def square():
    V = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    return V, construct_connectivity_data(V, [[0, 1, 2], [0, 2, 3]])


def test_second_heuristic_value_error_runs_fallback(monkeypatch):
    # a ValueError from the second heuristic (boundary prohibited) is a failure like
    # one from the first: the fallback runs instead of the error leaving the function
    calls = []

    def heuristic(*args, **kwargs):
        calls.append("heuristic")
        raise ValueError("No valid path found among closest candidates")

    monkeypatch.setattr(trace, "trace_initial_loop_with_heuristics", heuristic)
    monkeypatch.setattr(trace, "trace_initial_loop_fallback", lambda *args, **kwargs: calls.append("fallback"))
    V, conn = square()
    trace.trace_initial_loop_new(V, conn, PatchRegularSpaceData(), LoopTopologyData(), LoopGeometryData(), [],
                                 InitialLoopTracingData(), ignore_trace_geometry=True)
    assert calls == ["heuristic", "heuristic", "fallback"]


def test_non_initial_fallback_stops_when_trapped_loops_repeat(monkeypatch):
    # previous loop 1 is connected to the boundary, the retrace traps it again: with
    # nothing new to connect, the fallback raises instead of retracing forever
    prunes = []

    def prune(*args):
        prunes.append(1)
        if len(prunes) > 3:
            raise RuntimeError("the fallback retraces the same trapped loops forever")
        return False, [1]

    def trace_edge(V, conn, rsd, poly, edge_tracing_data, first_pair, second_pair, **kwargs):
        edge_tracing_data.traced_vids = [1, 2, 1]

    monkeypatch.setattr(trace, "regular_space_using_PRSD", lambda *args: None)
    monkeypatch.setattr(trace, "dijkstra_shortest_path_between_loops", lambda *args: [3, 0])
    monkeypatch.setattr(trace, "trace_one_feature_edge", trace_edge)
    monkeypatch.setattr(trace, "prune_after_subloop_closure", prune)
    V, conn = square()
    rsd = PatchRegularSpaceData(previous_loops=[[0, 1, 2], [3]])
    loop = LoopTopologyData(halfedges=[HalfEdge(5, True, False, 2, 2)])
    treatment = NonInitialLoopTreatmentData(loops_need_treatment=[1])
    with pytest.raises(ValueError, match=r"made no progress: trapped loops \[1\]"):
        trace.trace_non_initial_loop_fallback(V, conn, rsd, loop, LoopGeometryData({5: np.zeros((3, 3))}),
                                              LoopTracingData(), treatment, ignore_trace_geometry=True)
    assert len(prunes) == 1
