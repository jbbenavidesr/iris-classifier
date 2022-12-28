from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris_classifier.training import TrainingData

from iris_classifier.distances import (
    ChebyshevDistance,
    EuclidianDistance,
    ManhattanDistance,
    SorensenDistance,
)
from iris_classifier.hyperparameters import Hyperparameter
from iris_classifier.samples import Sample


def test_hyperparameter_init(training_data: TrainingData) -> None:
    """Test the initialization of a hyperparameter."""
    distance = EuclidianDistance()
    hyperparameter = Hyperparameter(3, distance, weakref.ref(training_data))
    assert hyperparameter.k == 3
    assert hyperparameter.algorithm == distance
    assert hyperparameter.data() == training_data


def test_hyperparameter_classify(training_data: TrainingData, sample: Sample) -> None:
    distance = EuclidianDistance()
    hyperparameter = Hyperparameter(3, distance, weakref.ref(training_data))
    assert hyperparameter.classify(sample) == "Iris-virginica"


def test_hyperparameter_test(training_data: TrainingData) -> None:
    """Test the testing of a hyperparameter."""

    distance = EuclidianDistance()
    hyperparameter = Hyperparameter(3, distance, weakref.ref(training_data))
    hyperparameter.test()
    assert hyperparameter.quality == 0.0


def test_euclidean_distance() -> None:
    """Test the euclidean distance."""
    ed = EuclidianDistance()
    sample_a = Sample(1, 2, 3, 4)
    sample_b = Sample(5, 6, 7, 8)
    assert ed.distance(sample_a, sample_b) == 8


def test_manhattan_distance() -> None:
    """Test the manhattan distance."""
    md = ManhattanDistance()
    sample_a = Sample(1, 2, 3, 4)
    sample_b = Sample(5, 6, 7, 8)
    assert md.distance(sample_a, sample_b) == 16


def test_chebyshev_distance() -> None:
    """Test the chebyshev distance."""
    cd = ChebyshevDistance()
    sample_a = Sample(1, 2, 3, 4)
    sample_b = Sample(5, 6, 7, 8)
    assert cd.distance(sample_a, sample_b) == 4


def test_sorensen_distance() -> None:
    """Test the sorensen distance."""
    sd = SorensenDistance()
    sample_a = Sample(1, 2, 3, 4)
    sample_b = Sample(5, 6, 7, 8)
    assert sd.distance(sample_a, sample_b) == 0.4444444444444444
