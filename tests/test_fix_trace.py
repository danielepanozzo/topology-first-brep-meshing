"""Regression tests for the fixes in operations/trace.py (initial and non-initial loop tracing)."""
import operations.trace as trace
from run_one_patch import construct_connectivity_data
from utils.data_structures import InitialLoopTracingData, LoopGeometryData, LoopTopologyData, PatchRegularSpaceData


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
