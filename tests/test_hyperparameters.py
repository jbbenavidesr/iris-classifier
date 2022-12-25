from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iris_classifier.training import TrainingData

from iris_classifier.hyperparameters import Hyperparameter


def test_hyperparameter_init(training_data: TrainingData) -> None:
    """Test the initialization of a hyperparameter."""
    hyperparameter = Hyperparameter(3, training_data)
    assert hyperparameter.k == 3
    assert hyperparameter.data() == training_data


def test_hyperparameter_classify(training_data: TrainingData) -> None:
    hyperparameter = Hyperparameter(3, training_data)
    assert hyperparameter.classify(training_data.training[0]) == "Iris-setosa"


def test_hyperparameter_test(training_data: TrainingData) -> None:
    """Test the testing of a hyperparameter."""
    iris_setosa_in_testing = 0
    for sample in training_data.testing:
        if sample.spicies == "Iris-setosa":
            iris_setosa_in_testing += 1

    quality = iris_setosa_in_testing / len(training_data.testing)
    hyperparameter = Hyperparameter(3, training_data)
    hyperparameter.test()
    assert hyperparameter.quality == quality
