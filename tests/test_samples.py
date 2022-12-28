from iris_classifier.samples import (
    ClassifiedSample,
    TestingKnownSample,
    TrainingKnownSample,
    UnknownSample,
)


def test_unknown_sample_init() -> None:
    """Test the initialization of an unknown and unclassified sample."""
    sample = UnknownSample(1.0, 2.0, 3.0, 4.0)
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0


def test_classified_sample_init() -> None:
    """Test the initialization of a classified sample."""
    classified_sample = ClassifiedSample(
        1.0,
        2.0,
        3.0,
        4.0,
        "Iris-setosa",
    )
    assert classified_sample.sepal_length == 1.0
    assert classified_sample.sepal_width == 2.0
    assert classified_sample.petal_length == 3.0
    assert classified_sample.petal_width == 4.0
    assert classified_sample.classification == "Iris-setosa"


def test_unknown_sample_repr_sample() -> None:
    """Test the repr of an unknown and unclassified sample."""
    sample = UnknownSample(1.0, 2.0, 3.0, 4.0)
    assert repr(sample) == (
        "UnknownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0)"
    )


def test_classified_sample_repr() -> None:
    """Test the repr of a classified sample."""
    classified_sample = ClassifiedSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert repr(classified_sample) == (
        "ClassifiedSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "classification='Iris-setosa')"
    )


def test_training_known_sample_init() -> None:
    """Test the initialization of a training known sample."""
    sample = TrainingKnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert sample.species == "Iris-setosa"
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0


def test_testing_known_sample_init() -> None:
    """Test the initialization of a testing known sample."""
    sample = TestingKnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.species == "Iris-setosa"
    assert sample.classification is None


def test_training_known_sample_repr() -> None:
    """Test the repr of a training known sample."""
    sample = TrainingKnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert repr(sample) == (
        "TrainingKnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species='Iris-setosa')"
    )


def test_testing_known_sample_unclassified_repr() -> None:
    """Test the repr of a testing known sample that has not been classified."""
    sample = TestingKnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert repr(sample) == (
        "TestingKnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species='Iris-setosa', "
        "classification=None)"
    )


def test_testing_known_sample_classified_repr() -> None:
    """Test the repr of a testing known sample that has been classified."""
    sample = TestingKnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa", "Iris-setosa")
    assert repr(sample) == (
        "TestingKnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species='Iris-setosa', "
        "classification='Iris-setosa')"
    )


def test_testing_known_sample_matches() -> None:
    """Test the matches method of a testing known sample."""
    sample = TestingKnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa", "Iris-setosa")
    assert sample.matches() is True


def test_testing_known_sample_does_not_match() -> None:
    """Test the matches method of a testing known sample."""
    sample = TestingKnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa", "Iris-versicolor")
    assert sample.matches() is False
