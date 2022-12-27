from iris_classifier.hyperparameters import Hyperparameter
from iris_classifier.training import TrainingData, ShufflingSamplePartition

from .mock_data import training_data as mock_training_data


def test_training_data_init() -> None:
    """Test the initialization of a training data."""
    training_data = TrainingData("Test training data")
    assert training_data.name == "Test training data"
    assert training_data.training == []
    assert training_data.testing == []
    assert training_data.tuning == []


def test_training_data_load() -> None:
    """Test the loading of a training data."""
    training_data = TrainingData("Test training data")
    training_data.load(
        mock_training_data, ShufflingSamplePartition(training_subset=0.9)
    )
    assert len(training_data.training) == 45
    assert len(training_data.testing) == 5
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


def test_training_data_classify(
    training_data: TrainingData, hyperparameter: Hyperparameter
) -> None:
    """Test the classification of a training data."""
    sample = training_data.classify(hyperparameter, training_data.testing[0])

    assert sample.classification == "Iris-virginica"


def test_shuffling_sample_partition_empty_init() -> None:
    """Test the initialization of a shuffling sample partition."""
    partition = ShufflingSamplePartition()
    assert partition.training == []
    assert partition.testing == []


def test_shuffling_sample_partition_init_with_iterable() -> None:
    """Test the loading of a shuffling sample partition."""
    partition = ShufflingSamplePartition(mock_training_data)
    assert len(partition.training) == 40
    assert len(partition.testing) == 10


def test_shuffling_sample_partition_training_subset_proportion() -> None:
    """Test the training subset proportion of a shuffling sample partition."""
    partition = ShufflingSamplePartition(mock_training_data, training_subset=0.9)
    assert len(partition.training) == 45
    assert len(partition.testing) == 5


def test_shuffling_sample_partition_training_subset_proportion_2() -> None:
    """Test the training subset proportion of a shuffling sample partition."""
    partition = ShufflingSamplePartition(mock_training_data, training_subset=0.5)
    assert len(partition.training) == 25
    assert len(partition.testing) == 25
