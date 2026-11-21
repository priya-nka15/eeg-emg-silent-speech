"""
URF/A-SSR Data Augmentation
PRD §29 — Split first, augment TRAIN only. Never augment validation or test.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


class AugmentationError(Exception):
    """Raised when augmentation is applied incorrectly."""


def _assert_train_only(split_tag: str) -> None:
    """Enforce that augmentation is only applied to train split."""
    if split_tag.lower() not in ("train", "training"):
        raise AugmentationError(
            f"Augmentation must only be applied to the TRAIN split, "
            f"not to '{split_tag}'. "
            "Split first, augment train only, never touch validation or test."
        )


def add_gaussian_noise(
    X: np.ndarray,
    snr_db: float = 20.0,
    seed: int = 42,
    split_tag: str = "train",
) -> np.ndarray:
    """
    Add Gaussian noise to signal windows.
    Only valid for training data.

    Args:
        X: Input array shape (n_windows, n_samples, n_channels) or
           (n_windows, n_features).
        snr_db: Signal-to-noise ratio in dB.
        seed: Random seed.
        split_tag: Must be 'train'.

    Returns:
        Augmented copy of X.
    """
    _assert_train_only(split_tag)
    rng = np.random.default_rng(seed)
    signal_power = np.mean(X ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), X.shape)
    return X + noise


def time_shift(
    X: np.ndarray,
    max_shift: int = 10,
    seed: int = 42,
    split_tag: str = "train",
) -> np.ndarray:
    """
    Apply a random time shift to temporal signal windows.
    Expects shape (n_windows, n_samples, n_channels).
    Only valid for training data.
    """
    _assert_train_only(split_tag)
    if X.ndim != 3:
        raise ValueError(
            f"time_shift expects 3D input (windows, samples, channels), got {X.ndim}D"
        )
    rng = np.random.default_rng(seed)
    shifts = rng.integers(-max_shift, max_shift + 1, size=X.shape[0])
    result = np.zeros_like(X)
    for i, shift in enumerate(shifts):
        result[i] = np.roll(X[i], shift, axis=0)
    return result


def amplitude_scale(
    X: np.ndarray,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    seed: int = 42,
    split_tag: str = "train",
) -> np.ndarray:
    """
    Apply per-window random amplitude scaling.
    Only valid for training data.
    """
    _assert_train_only(split_tag)
    rng = np.random.default_rng(seed)
    scales = rng.uniform(scale_range[0], scale_range[1], size=(X.shape[0],) + (1,) * (X.ndim - 1))
    return X * scales


def augment_train_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    methods: list = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation to training set.
    Returns augmented X and repeated labels.

    Each augmentation method produces one additional copy,
    appended to the original training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        methods: List of augmentation functions to apply. Defaults to [add_gaussian_noise].
        seed: Base seed; each method gets seed+i.

    Returns:
        (X_augmented, y_augmented) with originals included.
    """
    if methods is None:
        methods = [add_gaussian_noise]

    X_list = [X_train]
    y_list = [y_train]

    for i, method in enumerate(methods):
        X_aug = method(X_train, seed=seed + i, split_tag="train")
        X_list.append(X_aug)
        y_list.append(y_train)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)
