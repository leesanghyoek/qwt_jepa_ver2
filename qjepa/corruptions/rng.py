"""Seed tai lap duoc bang SHA-256 (spec muc 5.1).

Khong dung Python `hash()`: no randomize theo process nen khong tai lap duoc
giua cac lan chay.
"""

from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(
    master_seed: int,
    split: str,
    realization: int,
    trajectory: str,
    modality: str,
    frame_or_segment: int | str = 0,
) -> int:
    """Seed 63-bit on dinh cho mot (split, realization, trajectory, modality, don vi)."""
    raw = f"{master_seed}|{split}|{realization}|{trajectory}|{modality}|{frame_or_segment}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 1


def generator(
    master_seed: int,
    split: str,
    realization: int,
    trajectory: str,
    modality: str,
    frame_or_segment: int | str = 0,
) -> np.random.Generator:
    return np.random.default_rng(
        derive_seed(master_seed, split, realization, trajectory, modality, frame_or_segment)
    )
