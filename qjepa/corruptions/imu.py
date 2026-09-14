"""Suy giam IMU trong don vi vat ly (spec muc 5.3).

    u_bad[k] = u_clean[k] + b[k] + sigma_sample * eps[k] + spike[k]
    b[0]     ~ Uniform(-bias_bound, +bias_bound)
    b[k+1]   = b[k] + q_bias * sqrt(dt[k]) * eta[k]

Trace loi duoc sinh cho CA trajectory theo tung realization roi moi cat window,
nen hai cua so chong lan co cung gia tri nhieu tai cung timestamp. Khong reset
random walk o dau moi cua so (spec muc 5.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rng import generator


@dataclass(frozen=True)
class ImuCorruptionConfig:
    accel_noise_std: tuple[float, float] = (0.02, 0.10)
    gyro_noise_std: tuple[float, float] = (0.001, 0.005)
    bias_enabled: bool = False
    accel_bias_bound: float = 0.05
    gyro_bias_bound: float = 0.003
    drift_enabled: bool = False
    accel_bias_random_walk: float = 0.005
    gyro_bias_random_walk: float = 0.0002
    spike_enabled: bool = False
    spike_rate_per_second_per_triplet: float = 0.2
    accel_spike_amplitude: tuple[float, float] = (0.5, 2.0)
    gyro_spike_amplitude: tuple[float, float] = (0.02, 0.10)
    clean_probability: float = 0.1


class ImuCorruptor:
    """Sinh va cache trace loi theo (split, realization, trajectory)."""

    def __init__(
        self, config: ImuCorruptionConfig, master_seed: int = 42, cache_size: int = 8
    ) -> None:
        self.cfg = config
        self.master_seed = master_seed
        self.cache_size = cache_size
        self._cache: dict[tuple, tuple[np.ndarray, dict]] = {}

    def _build_trace(
        self, split: str, realization: int, trajectory: str, t: np.ndarray, force_clean: bool | None
    ) -> tuple[np.ndarray, dict]:
        cfg = self.cfg
        n = len(t)
        rng = generator(self.master_seed, split, realization, trajectory, "imu_trajectory")
        clean = bool(rng.random() < cfg.clean_probability)
        if force_clean is not None:
            clean = force_clean

        sigma = np.concatenate(
            [
                rng.uniform(*cfg.accel_noise_std, size=3),
                rng.uniform(*cfg.gyro_noise_std, size=3),
            ]
        )
        params = {"clean": clean, "sigma_sample": sigma.tolist()}
        if clean:
            return np.zeros((n, 6), dtype=np.float64), params

        error = rng.normal(0.0, 1.0, size=(n, 6)) * sigma[None, :]

        if cfg.bias_enabled or cfg.drift_enabled:
            bound = np.array([cfg.accel_bias_bound] * 3 + [cfg.gyro_bias_bound] * 3)
            b0 = rng.uniform(-bound, bound) if cfg.bias_enabled else np.zeros(6)
            bias = np.repeat(b0[None, :], n, axis=0)
            if cfg.drift_enabled:
                q = np.array(
                    [cfg.accel_bias_random_walk] * 3 + [cfg.gyro_bias_random_walk] * 3
                )
                dt = np.diff(t)
                steps = rng.normal(0.0, 1.0, size=(n - 1, 6)) * q[None, :] * np.sqrt(dt)[:, None]
                bias = b0[None, :] + np.concatenate(
                    [np.zeros((1, 6)), np.cumsum(steps, axis=0)], axis=0
                )
            error += bias
            params["bias_initial"] = np.asarray(b0).tolist()

        if cfg.spike_enabled:
            dt = np.concatenate([[np.median(np.diff(t))], np.diff(t)])
            p_event = 1.0 - np.exp(-cfg.spike_rate_per_second_per_triplet * dt)
            for triplet, (lo, hi) in enumerate(
                (cfg.accel_spike_amplitude, cfg.gyro_spike_amplitude)
            ):
                hits = rng.random(n) < p_event
                idx = np.flatnonzero(hits)
                if idx.size:
                    axis = rng.integers(0, 3, size=idx.size) + 3 * triplet
                    sign = rng.choice((-1.0, 1.0), size=idx.size)
                    amp = rng.uniform(lo, hi, size=idx.size) * sign
                    error[idx, axis] += amp
            params["spike_events"] = int(np.count_nonzero(hits))

        return error, params

    def trace(
        self,
        split: str,
        realization: int,
        trajectory: str,
        t: np.ndarray,
        force_clean: bool | None = None,
    ) -> tuple[np.ndarray, dict]:
        key = (split, realization, trajectory, force_clean)
        if key not in self._cache:
            if len(self._cache) >= self.cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = self._build_trace(split, realization, trajectory, t, force_clean)
        return self._cache[key]

    def __call__(
        self,
        imu_clean_phys: np.ndarray,
        *,
        split: str,
        realization: int,
        trajectory: str,
        times: np.ndarray,
        start: int,
        end: int,
        force_clean: bool | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Cat window [start:end] tu trace cua ca trajectory roi cong vao clean."""
        error, params = self.trace(split, realization, trajectory, times, force_clean)
        return (imu_clean_phys + error[start:end]).astype(np.float64), params
