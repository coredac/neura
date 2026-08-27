"""Small regression tests for the budgeted CP-SAT mapper.

The test file is intentionally runnable in source checkouts that do not carry
the optional OR-Tools wheel: source-level contract tests still run and solver
tests are skipped with a useful reason.
"""

import importlib.util
from pathlib import Path

import pytest


MAPPER_PATH = Path(__file__).with_name("exact_mapper_cpsat.py")


def load_mapper():
    """Import the script as a module, or skip when OR-Tools is unavailable."""
    pytest.importorskip("ortools", reason="CP-SAT regression tests need OR-Tools")
    spec = importlib.util.spec_from_file_location("exact_mapper_cpsat",
                                                  MAPPER_PATH)
    mapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapper)
    return mapper


def make_arch(*, regs=1, regfiles=1, fu_classes=None):
    return {
        "tiles": [{"id": 0, "x": 0, "y": 0, "regs": regs,
                   "regfiles": regfiles}],
        "links": [],
        "fu_class_tiles": fu_classes or {"pipe": [0], "single": [0]},
        "num_tiles": 1,
        "ctrl_mem_items": 8,
    }


def make_data(ops, *, edges=None, arch=None):
    return {"ops": ops, "edges": edges or [], "arch": arch or make_arch()}


def test_source_uses_a_deterministic_budget_and_honest_verdict_names():
    source = MAPPER_PATH.read_text()
    assert "max_deterministic_time" in source
    assert "max_time_in_seconds" not in source
    assert "TRUE_MIN_II" not in source
    assert "proven optimal" not in source
    assert "dest=\"mr\"" not in source


def test_pipeline_start_end_overlap_is_allowed():
    mapper = load_mapper()
    data = make_data([{"class": "pipe", "latency": 2},
                      {"class": "pipe", "latency": 2}])
    result = mapper.schedule(data, 2, 100.0, mapper.shortest_hops(data["arch"]))
    assert isinstance(result, dict)


def test_pipeline_in_stages_can_overlap_and_latency_can_cross_ii():
    mapper = load_mapper()
    data = make_data([{"class": "pipe", "latency": 4} for _ in range(3)])
    result = mapper.schedule(data, 3, 100.0, mapper.shortest_hops(data["arch"]))
    assert isinstance(result, dict)


def test_single_cycle_still_conflicts_with_every_pipeline_stage():
    mapper = load_mapper()
    data = make_data([{"class": "pipe", "latency": 2},
                      {"class": "single", "latency": 1}])
    result = mapper.schedule(data, 2, 100.0, mapper.shortest_hops(data["arch"]))
    assert result is None


def test_direct_same_tile_route_consumes_a_register_slot():
    mapper = load_mapper()
    data = make_data([{"class": "pipe", "latency": 1},
                      {"class": "pipe", "latency": 1}],
                     edges=[{"s": 0, "d": 1, "w": 0}])
    sched = {0: (0, 0), 1: (0, 1)}
    feasible, routes, status = mapper.route(data, sched, 1, 100.0,
                                            want_routes=True)
    assert feasible
    assert status == "feasible"
    assert routes[(0, 1)] == [(0, 1)]


def test_direct_same_tile_route_rejects_zero_register_capacity():
    mapper = load_mapper()
    arch = make_arch(regs=0, regfiles=0, fu_classes={"pipe": [0]})
    data = make_data([{"class": "pipe", "latency": 1},
                      {"class": "pipe", "latency": 1}],
                     edges=[{"s": 0, "d": 1, "w": 0}], arch=arch)
    sched = {0: (0, 0), 1: (0, 1)}
    feasible, routes, status = mapper.route(data, sched, 1, 100.0,
                                            want_routes=True)
    assert not feasible
    assert routes is None
    assert status == mapper.INFEASIBLE


def test_independent_multi_cycle_holds_share_storage_but_need_write_ports():
    mapper = load_mapper()
    arch = make_arch(regs=2, regfiles=1, fu_classes={"pipe": [0]})
    data = make_data(
        [{"class": "pipe", "latency": 1} for _ in range(4)],
        edges=[{"s": 0, "d": 1, "w": 0},
               {"s": 2, "d": 3, "w": 0}],
        arch=arch)
    # Both values hold from ready cycle 1 through consume cycle 2. Two
    # registers are enough for storage, but both runs start and end in the
    # same modulo slot, so one register file cannot provide both ports.
    sched = {0: (0, 0), 1: (0, 2), 2: (0, 0), 3: (0, 2)}
    feasible, routes, status = mapper.route(data, sched, 1, 100.0,
                                            want_routes=True)
    assert not feasible
    assert routes is None
    assert status == mapper.INFEASIBLE
