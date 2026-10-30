from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .param_space import Categorical, Int, ParamSpace, Real, _active_assignment


class Sampler(Protocol):
    """
    Interface for configuration samplers used by tuners.

    A sampler knows how to draw new configurations (dicts) from a given
    ParamSpace, possibly using a learned model of 'good regions' in the
    hyperparameter space.
    """

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """
        Sample a new configuration.

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator to use.

        Returns
        -------
        dict[str, Any]
            Configuration dictionary mapping parameter names to values.
        """
        ...


@dataclass
class UniformSampler:
    """
    Uniform sampler that delegates to ParamSpace.sample.

    This reproduces the current behavior of the base Tuner and RacingTuner.
    """

    param_space: ParamSpace

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return self.param_space.sample(rng)


@dataclass
class ModelBasedSampler:
    """
    Model-based sampler with two exploitation modes:

    1) Marginal model (independent per parameter):
       - Categorical: empirical frequencies
       - Real/Int: Gaussian in linear/log space

    2) Joint local model (interaction-preserving):
       - Sample an anchor from observed good configurations
       - Apply small, bounded perturbations around that anchor

    The joint mode keeps cross-parameter structure from successful configs,
    while marginals provide broader coverage.
    """

    param_space: ParamSpace
    exploration_prob: float = 0.2
    """
    Probability of ignoring the learned model and sampling uniformly from the
    ParamSpace. This ensures persistent exploration.
    """

    min_samples_to_model: int = 5
    """
    Minimum number of 'good' configurations required before using the learned
    marginal model for exploitation sampling.
    """

    _cat_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    _real_models: dict[str, dict[str, float]] = field(default_factory=dict)
    _int_models: dict[str, dict[str, float]] = field(default_factory=dict)
    _joint_pool: list[dict[str, Any]] = field(default_factory=list)
    joint_sampling_prob: float = 0.7
    min_samples_for_joint: int = 8
    neighbor_noise_scale: float = 0.2

    def update(self, good_configs: Sequence[Mapping[str, Any]]) -> None:
        """
        Update the internal marginal model from a collection of 'good' configs.

        If the number of good_configs is below min_samples_to_model, the internal
        model is cleared and only uniform sampling will be used.
        """
        if len(good_configs) < self.min_samples_to_model:
            self._cat_models.clear()
            self._real_models.clear()
            self._int_models.clear()
            self._joint_pool.clear()
            return

        observed: dict[str, list[Any]] = {name: [] for name in self.param_space.params.keys()}

        for cfg in good_configs:
            for name in observed.keys():
                if name in cfg:
                    observed[name].append(cfg[name])

        self._cat_models.clear()
        self._real_models.clear()
        self._int_models.clear()
        self._joint_pool.clear()

        for cfg in good_configs:
            cfg_dict = dict(cfg)
            try:
                self.param_space.validate(cfg_dict)
            except ValueError:
                continue
            active_cfg = _active_assignment(self.param_space, cfg_dict)
            if active_cfg:
                self._joint_pool.append(active_cfg)

        for name, spec in self.param_space.params.items():
            values = observed.get(name, [])
            if not values:
                continue

            if isinstance(spec, Categorical):
                counts: dict[Any, int] = {}
                for v in values:
                    counts[v] = counts.get(v, 0) + 1

                total = sum(counts.values())
                if total == 0:
                    continue

                choices: list[Any] = []
                probs: list[float] = []
                for choice in spec.choices:
                    c = counts.get(choice, 0)
                    if c > 0:
                        choices.append(choice)
                        probs.append(c / total)

                if not choices:
                    continue

                self._cat_models[name] = {
                    "choices": choices,
                    "probs": np.asarray(probs, dtype=float),
                }

            elif isinstance(spec, Real):
                arr = np.array(values, dtype=float)
                is_log = spec.log
                if is_log:
                    # For log params, we model the distribution in log-space
                    # Values are guaranteed > 0 by ParamSpace validation
                    arr = np.log(arr)

                mean = float(np.mean(arr))
                std = float(np.std(arr))
                # Ensure non-zero std for sampling
                if std < 1e-9:
                    if is_log:
                        # 10% of the log-range
                        log_range = math.log(spec.high) - math.log(spec.low)
                        std = 0.1 * log_range
                    else:
                        std = 0.1 * (spec.high - spec.low)

                self._real_models[name] = {
                    "mean": mean,
                    "std": std,
                    "low": spec.low,
                    "high": spec.high,
                    "log": is_log,
                }

            elif isinstance(spec, Int):
                arr = np.array(values, dtype=float)
                is_log = spec.log
                if is_log:
                    arr = np.log(arr)

                mean = float(np.mean(arr))
                std = float(np.std(arr))
                if std < 1e-9:
                    if is_log:
                        log_range = math.log(spec.high) - math.log(spec.low)
                        std = max(0.1, 0.1 * log_range)
                    else:
                        std = max(1.0, 0.1 * (spec.high - spec.low))

                self._int_models[name] = {
                    "mean": mean,
                    "std": std,
                    "low": spec.low,
                    "high": spec.high,
                    "log": is_log,
                }

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """
        Sample a new configuration using a mixture of uniform exploration and
        marginal exploitation. If insufficient model data exist, falls back to
        uniform sampling.
        """
        if rng.random() < self.exploration_prob or (not self._cat_models and not self._real_models and not self._int_models):
            return self.param_space.sample(rng)

        if len(self._joint_pool) >= self.min_samples_for_joint and rng.random() < self.joint_sampling_prob:
            joint_cfg = self._sample_joint(rng)
            if joint_cfg is not None:
                return joint_cfg

        return self._sample_marginal(rng)

    def _sample_marginal(self, rng: np.random.Generator) -> dict[str, Any]:
        cfg: dict[str, Any] = {}

        for name, spec in self.param_space.params.items():
            if isinstance(spec, Categorical) and name in self._cat_models:
                model = self._cat_models[name]
                choices = model["choices"]
                probs = model["probs"]
                idx = int(rng.choice(len(choices), p=probs))
                cfg[name] = choices[idx]
                continue

            if isinstance(spec, Real) and name in self._real_models:
                m = self._real_models[name]
                # Sample from Normal(mean, std), clipped to bounds
                val = rng.normal(m["mean"], m["std"])
                if m.get("log", False):
                    val = math.exp(val)
                val = max(float(m["low"]), min(float(m["high"]), val))
                cfg[name] = val
                continue

            if isinstance(spec, Int) and name in self._int_models:
                m = self._int_models[name]
                val = rng.normal(m["mean"], m["std"])
                if m.get("log", False):
                    val = math.exp(val)
                val = int(round(val))
                val = max(int(m["low"]), min(int(m["high"]), val))
                cfg[name] = val
                continue

            tmp = self.param_space.sample(rng)
            if name in tmp:
                cfg[name] = tmp[name]

        active_cfg = _active_assignment(self.param_space, cfg)
        try:
            self.param_space.validate(active_cfg)
        except ValueError:
            return self.param_space.sample(rng)
        return active_cfg

    def _sample_joint(self, rng: np.random.Generator) -> dict[str, Any] | None:
        if not self._joint_pool:
            return None

        anchor = dict(self._joint_pool[int(rng.integers(0, len(self._joint_pool)))])
        cfg: dict[str, Any] = dict(anchor)

        for name, spec in self.param_space.params.items():
            if not self.param_space.is_active(name, cfg):
                continue

            if isinstance(spec, Categorical):
                model = self._cat_models.get(name)
                # Keep anchor category most of the time to preserve interactions.
                if model is not None and rng.random() < self.neighbor_noise_scale:
                    choices = model["choices"]
                    probs = model["probs"]
                    idx = int(rng.choice(len(choices), p=probs))
                    cfg[name] = choices[idx]
                elif name not in cfg:
                    idx = int(rng.integers(0, len(spec.choices)))
                    cfg[name] = spec.choices[idx]
                continue

            if isinstance(spec, Real):
                m = self._real_models.get(name)
                base = float(cfg.get(name, 0.5 * (spec.low + spec.high)))
                if m is not None:
                    std = max(float(m["std"]) * self.neighbor_noise_scale, 1e-12)
                    if m.get("log", False):
                        base_log = math.log(max(base, 1e-12))
                        val = float(math.exp(rng.normal(base_log, std)))
                    else:
                        val = float(rng.normal(base, std))
                else:
                    range_width = float(spec.high - spec.low)
                    val = float(rng.normal(base, self.neighbor_noise_scale * max(range_width, 1e-12)))
                cfg[name] = max(float(spec.low), min(float(spec.high), val))
                continue

            if isinstance(spec, Int):
                m = self._int_models.get(name)
                base = int(cfg.get(name, int(round(0.5 * (spec.low + spec.high)))))
                if m is not None:
                    std = max(float(m["std"]) * self.neighbor_noise_scale, 1e-12)
                    if m.get("log", False):
                        base_log = math.log(max(float(base), 1e-12))
                        val = int(round(math.exp(rng.normal(base_log, std))))
                    else:
                        val = int(round(rng.normal(base, std)))
                else:
                    range_width = max(1, int(spec.high - spec.low))
                    val = int(round(rng.normal(base, self.neighbor_noise_scale * range_width)))
                cfg[name] = max(int(spec.low), min(int(spec.high), val))
                continue

        active_cfg = _active_assignment(self.param_space, cfg)
        try:
            self.param_space.validate(active_cfg)
        except ValueError:
            return None
        return active_cfg


__all__ = ["Sampler", "UniformSampler", "ModelBasedSampler"]
