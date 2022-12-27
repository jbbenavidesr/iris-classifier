import pytest

from iris_classifier.samples import KnownSample, Purpose, Sample, Species, UnknownSample


def test_sample_init() -> None:
    """Test the initialization of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0


def test_sample_repr_sample() -> None:
    """Test the repr of an unknown and unclassified sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    assert repr(sample) == (
        "Sample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0)"
    )


def test_testing_known_sample_init() -> None:
    """Test the initialization of a known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING, "Iris-setosa")
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.purpose == Purpose.TESTING
    assert sample.species == "Iris-setosa"


def test_training_known_sample_init() -> None:
    """Test the initialization of a known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING, "Iris-setosa")
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.purpose == Purpose.TRAINING
    assert sample.species == "Iris-setosa"


def test_known_sample_raises_error_on_other_purpose() -> None:
    """Test the initialization of a known sample with a purpose other than
    training or testing."""
    with pytest.raises(ValueError):
        KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.CLASSIFICATION, "Iris-setosa")


def test_training_known_sample_repr() -> None:
    """Test the repr of a training known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING.value, "Iris-setosa")
    assert repr(sample) == (
        "KnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        f"purpose={Purpose.TRAINING.value}, "
        "species=Iris-setosa)"
    )


def test_training_known_sample_raises_error_on_getting_classification() -> None:
    """Test the initialization of a known sample with a purpose other than
    training or testing."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING.value, "Iris-setosa")
    with pytest.raises(AttributeError):
        sample.classification


def test_training_known_sample_raises_error_on_setting_classification() -> None:
    """Test the initialization of a known sample with a purpose other than
    training or testing."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING.value, "Iris-setosa")
    with pytest.raises(AttributeError):
        sample.classification = Species.SETOSA


def test_testing_known_sample_has_classification() -> None:
    """Test the initialization of a known sample with a purpose other than
    training or testing."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING.value, "Iris-setosa")
    assert sample.classification is None
    sample.classification = Species.SETOSA
    assert sample.classification == "Iris-setosa"


def test_testing_known_sample_unclassified_repr() -> None:
    """Test the repr of an unclassified known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING.value, "Iris-setosa")
    assert repr(sample) == (
        "KnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        f"purpose={Purpose.TESTING.value}, "
        "species=Iris-setosa)"
    )


def test_testing_known_sample_classified_repr() -> None:
    """Test the repr of a classified known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING.value, "Iris-setosa")
    sample.classification = Species.SETOSA
    assert repr(sample) == (
        "KnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        f"purpose={Purpose.TESTING.value}, "
        "species=Iris-setosa, "
        "classification=Iris-setosa)"
    )


def test_matches_raises_error_on_training_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING, "Iris-setosa")
    with pytest.raises(AttributeError):
        sample.matches()


def test_matches_of_correctly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING, "Iris-setosa")
    sample.classification = Species.SETOSA
    assert sample.matches()


def test_matches_of_uncorrectly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING, "Iris-setosa")
    sample.classification = Species.VERSICOLOR
    assert not sample.matches()


def test_unknown_sample_init() -> None:
    """Test the initialization of an unknown sample."""
    sample = UnknownSample(1.0, 2.0, 3.0, 4.0)
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0


def test_unknown_sample_repr() -> None:
    """Test the repr of an unknown sample."""
    sample = UnknownSample(1.0, 2.0, 3.0, 4.0)
    assert repr(sample) == (
        "UnknownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "classification=None)"
    )
