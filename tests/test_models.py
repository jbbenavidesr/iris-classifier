import pytest

from iris_classifier.models import Hyperparameter, Sample, TrainingData

from .mock_data import training_data as mock_training_data


def test_sample_init() -> None:
    """Test the initialization of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.spicies == "Iris-setosa"
    assert sample.classification is None


def test_sample_repr_of_unknown_unclassified_sample() -> None:
    """Test the repr of an unknown and unclassified sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    assert repr(sample) == (
        "UnknownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "spicies=None)"
    )


def test_sample_repr_of_known_unclassified_sample() -> None:
    """Test the repr of a known and unclassified sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert repr(sample) == (
        "KnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "spicies='Iris-setosa')"
    )


def test_sample_classify() -> None:
    """Test the classification of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    sample.classify("Iris-setosa")
    assert sample.classification == "Iris-setosa"


def test_sample_repr_of_unknown_classified_sample() -> None:
    """Test the repr of an unknown and classified sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    sample.classify("Iris-setosa")
    assert repr(sample) == (
        "UnknownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "spicies=None, "
        "classification=Iris-setosa)"
    )


def test_sample_repr_of_known_classified_sample() -> None:
    """Test the repr of a known and classified sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    sample.classify("Iris-setosa")
    assert repr(sample) == (
        "KnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "spicies='Iris-setosa', "
        "classification=Iris-setosa)"
    )


def test_matches_of_unclassified_sample() -> None:
    """Test the matching of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert not sample.matches()


def test_matches_of_correctly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    sample.classify("Iris-setosa")
    assert sample.matches()


def test_matches_of_uncorrectly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    sample.classify("Iris-versicolor")
    assert not sample.matches()


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
    training_data.load(mock_training_data)
    assert len(training_data.training) == 45
    assert len(training_data.testing) == 5
    assert training_data.uploaded is not None


@pytest.fixture
def training_data() -> TrainingData:
    """Return a training data."""
    training_data = TrainingData("Test training data")
    training_data.load(mock_training_data)
    return training_data


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


@pytest.fixture
def hyperparameter(training_data: TrainingData) -> Hyperparameter:
    """Return a hyperparameter."""
    return Hyperparameter(3, training_data)


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
    sample = training_data.classify(hyperparameter, training_data.training[0])

    assert sample.classification == "Iris-setosa"
