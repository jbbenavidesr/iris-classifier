"""Module containing the classifiers used to classify the iris dataset"""
from typing import NamedTuple, Callable, TYPE_CHECKING, Counter, cast
import collections
import bisect
import heapq

if TYPE_CHECKING:
    from .distances import DistanceFunc

from .models import TrainingKnownSample, TrainingList, AnySample
from .distances import DistanceFunc

Classifier = Callable[[int, DistanceFunc, TrainingList, AnySample], str]


class Measured(NamedTuple):
    """Utility class used to store distances during classification"""

    distance: float
    sample: TrainingKnownSample


def k_nn_1(
    k: int, dist: DistanceFunc, training_data: TrainingList, unknown: AnySample
) -> str:
    """Classify a sample using the most naive implementation of the k-NN algorithm.

    It works by sorting the distances and then taking the most common species in the
    k nearest neighbors.

    :param k: The number of nearest neighbors to consider.
    :param dist: The distance function to use.
    :param training_data: The training data to use.
    :param unknown: The sample to classify.
    :return: The classification of the sample.
    """
    measured = sorted((Measured(dist(s, unknown), s) for s in training_data))
    k_nearest = measured[:k]
    k_frequencies: Counter[str] = collections.Counter(
        s.sample.sample.species for s in k_nearest
    )
    mode, _ = k_frequencies.most_common(1)[0]
    return mode


def k_nn_b(
    k: int, dist: DistanceFunc, training_data: TrainingList, unknown: AnySample
) -> str:
    """Classify a sample using a bisection strategy for the k-NN algorithm.

    It uses the bisect module to insert the distances in the correct position in the
    sorted list of k distances and then takes the most common species.

    :param k: The number of nearest neighbors to consider.
    :param dist: The distance function to use.
    :param training_data: The training data to use.
    :param unknown: The sample to classify.
    :return: The classification of the sample.
    """
    k_nearest = [
        Measured(float("inf"), cast(TrainingKnownSample, None)) for _ in range(k)
    ]
    for t in training_data:
        t_dist = dist(t, unknown)
        if t_dist > k_nearest[-1].distance:
            continue
        new = Measured(t_dist, t)
        k_nearest.insert(bisect.bisect_left(k_nearest, new), new)
        k_nearest.pop()
    k_frequencies: Counter[str] = collections.Counter(
        s.sample.sample.species for s in k_nearest
    )
    mode, _ = k_frequencies.most_common(1)[0]
    return mode


def k_nn_q(
    k: int, dist: DistanceFunc, training_data: TrainingList, unknown: AnySample
) -> str:
    """Classify a sample using a heap strategy for the k-NN algorithm.

    It uses the heapq module to insert the distances in the correct position in the
    sorted list of k distances and then takes the most common species.

    :param k: The number of nearest neighbors to consider.
    :param dist: The distance function to use.
    :param training_data: The training data to use.
    :param unknown: The sample to classify.
    :return: The classification of the sample.
    """
    measured_iter = (Measured(dist(s, unknown), s) for s in training_data)
    k_nearest = heapq.nsmallest(k, measured_iter)
    k_frequencies: Counter[str] = collections.Counter(
        s.sample.sample.species for s in k_nearest
    )
    mode, _ = k_frequencies.most_common(1)[0]
    return mode
