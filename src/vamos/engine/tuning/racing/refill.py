from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .param_space import Categorical, Int, ParamSpace, ParamType, Real, _dependency_order
from .sampler import Sampler
from .state import ConfigState, EliteEntry

if TYPE_CHECKING:
    from .scenario import Scenario


def make_neighbor_config(base_config: dict[str, Any], param_space: ParamSpace, rng: np.random.Generator) -> dict[str, Any]:
    """Create a new configuration by applying small perturbations to a base configuration.

    Parameters are visited parents first. An active parameter that the base sets is
    perturbed, and one it leaves out, because it was inactive there, is drawn from its
    own range. A parameter whose conditions fail after the perturbation is dropped with
    its descendants, as in ``ParamSpace.sample``. Keys outside the space are kept.
    """
    kept: dict[str, Any] = {}
    for name in _dependency_order(param_space):
        if not param_space.is_active(name, kept):
            continue
        spec = param_space.params[name]
        kept[name] = _perturb(spec, base_config[name], rng) if name in base_config else spec.sample(rng)

    cfg = {name: kept[name] for name in param_space.params if name in kept}
    cfg.update({name: value for name, value in base_config.items() if name not in param_space.params})
    return cfg


def _perturb(spec: ParamType, current_value: Any, rng: np.random.Generator) -> Any:
    """A value near ``current_value`` within the parameter's range."""
    if isinstance(spec, Real):
        low = spec.low
        high = spec.high

        if spec.log:
            log_low = math.log(low)
            log_high = math.log(high)
            log_range = log_high - log_low
            sigma = 0.1 * log_range

            # Perturb in log-space
            cur_log = math.log(float(current_value))
            new_log = cur_log + float(rng.normal(0.0, sigma))
            new_val = math.exp(new_log)
        else:
            range_width = high - low
            if range_width <= 0:
                return float(current_value)
            sigma = 0.1 * range_width
            new_val = float(current_value) + float(rng.normal(0.0, sigma))

        return max(low, min(high, new_val))
    if isinstance(spec, Int):
        low = spec.low
        high = spec.high

        if spec.log:
            log_low = math.log(low)
            log_high = math.log(high)
            log_range = log_high - log_low
            sigma = max(0.1, 0.1 * log_range)  # ensure some variance

            cur_log = math.log(float(current_value))
            new_log = cur_log + float(rng.normal(0.0, sigma))
            new_int = int(round(math.exp(new_log)))
        else:
            range_width = max(1, high - low)
            step = max(1, int(round(0.1 * range_width)))
            delta = int(rng.integers(-step, step + 1))
            new_int = int(current_value) + delta

        return max(low, min(high, new_int))
    if isinstance(spec, Categorical):
        choices = list(spec.choices)
        if not choices:
            return current_value
        keep_prob = 0.7
        if current_value in choices and rng.random() < keep_prob:
            return current_value
        available = [c for c in choices if c != current_value] or choices
        return available[int(rng.integers(0, len(available)))]
    return current_value


def refill_population(
    configs: list[ConfigState],
    *,
    scenario: Scenario,
    param_space: ParamSpace,
    sampler: Sampler,
    elite_archive: list[EliteEntry],
    target_population_size: int,
    rng: np.random.Generator,
    next_config_id: int,
    max_new: int | None = None,
) -> int:
    """
    Spawn new configurations when elitist restarts are enabled and alive count is below target.
    At most ``max_new`` are spawned when it is given. Returns the updated next_config_id.
    """
    alive_states = [c for c in configs if c.alive]
    n_alive = len(alive_states)

    if n_alive >= target_population_size:
        return next_config_id

    n_to_spawn = target_population_size - n_alive
    if max_new is not None:
        n_to_spawn = min(n_to_spawn, max_new)
    if n_to_spawn <= 0:
        return next_config_id

    n_neighbors = int(round(scenario.neighbor_fraction * n_to_spawn))
    n_neighbors = max(0, min(n_neighbors, n_to_spawn))
    n_fresh = n_to_spawn - n_neighbors

    elite_configs: list[dict[str, Any]] = [e.config for e in elite_archive]
    if not elite_configs:
        elite_configs = [c.config for c in alive_states]

    for _ in range(n_neighbors):
        base_cfg = elite_configs[int(rng.integers(0, len(elite_configs)))]
        neighbor_cfg = make_neighbor_config(base_cfg, param_space, rng)
        state = ConfigState(
            config_id=next_config_id,
            config=neighbor_cfg,
            alive=True,
        )
        next_config_id += 1
        configs.append(state)

    for _ in range(n_fresh):
        cfg = sampler.sample(rng)
        state = ConfigState(
            config_id=next_config_id,
            config=cfg,
            alive=True,
        )
        next_config_id += 1
        configs.append(state)

    return next_config_id


__all__ = ["make_neighbor_config", "refill_population"]
