from iris_classifier.hyperparameters import Hyperparameter
from iris_classifier.samples import KnownSample
from iris_classifier.training import TrainingData


def test_training_data_init() -> None:
    """Test the initialization of a training data."""
    training_data = TrainingData("Test training data")
    assert training_data.name == "Test training data"
    assert training_data.training == []
    assert training_data.testing == []
    assert training_data.tuning == []


def test_training_data_load(mock_training_data: list[KnownSample]) -> None:
    """Test the loading of a training data."""
    training_data = TrainingData("Test training data")
    training_data.load(mock_training_data)
    assert len(training_data.training) == 27
    assert len(training_data.testing) == 3
    assert training_data.uploaded is not None


def test_training_data_test(
    training_data: TrainingData, hyperparameter: Hyperparameter
) -> None:
    """Test the testing of a training data."""
    training_data.test(hyperparameter)

    assert len(training_data.tuning) == 1
    assert training_data.tuning[0] is hyperparameter
    assert hyperparameter.quality is not None
    assert training_data.tested is not None
