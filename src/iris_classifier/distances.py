"""Different Implementations of pluggable distance functions for the KNN."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

if TYPE_CHECKING:
    from .models import TrainingKnownSample, AnySample

DistanceFunc = Callable[[TrainingKnownSample, AnySample], float]


def minkowski(
    s1: TrainingKnownSample,
    s2: AnySample,
    m: int,
    summarize: Callable[[Iterable[float]], float] = sum,
) -> float:
    """Generalization of distance calculations"""
    return (
        summarize((abs(a - b) ** m for a, b in zip(s1.sample.sample, s2.sample)))
    ) ** (1 / m)


def manhattan(s1: TrainingKnownSample, s2: AnySample) -> float:
    """Manhattan distance"""
    return minkowski(s1, s2, m=1)


def euclidean(s1: TrainingKnownSample, s2: AnySample) -> float:
    """Euclidean distance"""
    return minkowski(s1, s2, m=2)


def chebyshev(s1: TrainingKnownSample, s2: AnySample) -> float:
    """Chebyshev distance"""
    return minkowski(s1, s2, m=1, summarize=max)
