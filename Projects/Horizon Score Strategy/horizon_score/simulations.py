"""Explicit strategy/benchmark row alignment for simulation filtering."""


def align_simulation_results(strategies, benchmarks, key="Simulation ID"):
    for frame in (strategies, benchmarks):
        if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Expected unique non-missing {key}")
    if set(strategies[key]) != set(benchmarks[key]):
        raise ValueError("Strategy and benchmark simulation IDs differ")
    strategies = strategies.reset_index(drop=True).copy()
    benchmarks = benchmarks.set_index(key).loc[strategies[key]].reset_index()
    return strategies, benchmarks
