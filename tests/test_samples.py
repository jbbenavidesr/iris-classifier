import pytest

from iris_classifier.exceptions import InvalidSampleError, OutOfBoundsError
from iris_classifier.samples import (
    KnownSample,
    Sample,
    Purpose,
    UnknownSample,
)


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


def test_sample_from_dict() -> None:
    """Test the creation of a sample from a dictionary."""
    sample = Sample.from_dict(
        {
            "sepal_length": "1.0",
            "sepal_width": "2.0",
            "petal_length": "3.0",
            "petal_width": "4.0",
        }
    )
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0


def test_sample_from_dict_invalid() -> None:
    """Test the creation of a sample from a dictionary with invalid values."""
    with pytest.raises(InvalidSampleError):
        Sample.from_dict(
            {
                "sepal_length": "1.0",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "invalid",
            }
        )


def test_sample_from_dict_out_of_bounds() -> None:
    """Test the creation of a sample from a dictionary with out of bounds values."""
    with pytest.raises(OutOfBoundsError):
        Sample.from_dict(
            {
                "sepal_length": "1.0",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "-3.0",
            }
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
    """Test the initialization of a known sample with a purpose other than training or testing."""
    with pytest.raises(ValueError):
        KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.CLASSIFICATION, "Iris-setosa")


def test_training_known_sample_repr() -> None:
    """Test the repr of a training known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING.value, "Iris-setosa")
    assert repr(sample) == (
        "TrainingKnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species=Iris-setosa)"
    )


def test_training_known_sample_raises_error_on_getting_classification() -> None:
    """Test the initialization of a known sample with a purpose other than training or testing."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING.value, "Iris-setosa")
    with pytest.raises(AttributeError):
        sample.classification


def test_training_known_sample_raises_error_on_setting_classification() -> None:
    """Test the initialization of a known sample with a purpose other than training or testing."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING.value, "Iris-setosa")
    with pytest.raises(AttributeError):
        sample.classification = "Iris-setosa"


def test_testing_known_sample_has_classification() -> None:
    """Test the initialization of a known sample with a purpose other than training or testing."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING.value, "Iris-setosa")
    assert sample.classification is None
    sample.classification = "Iris-setosa"
    assert sample.classification == "Iris-setosa"


def test_testing_known_sample_unclassified_repr() -> None:
    """Test the repr of an unclassified known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING.value, "Iris-setosa")
    assert repr(sample) == (
        "TestingKnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species=Iris-setosa)"
    )


def test_testing_known_sample_classified_repr() -> None:
    """Test the repr of a classified known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING.value, "Iris-setosa")
    sample.classification = "Iris-setosa"
    assert repr(sample) == (
        "TestingKnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species=Iris-setosa, "
        "classification=Iris-setosa)"
    )


def test_training_known_sample_from_dict() -> None:
    """Test the creation of a known sample from a dict."""
    sample = KnownSample.from_dict(
        {
            "sepal_length": "1.0",
            "sepal_width": "2.0",
            "petal_length": "3.0",
            "petal_width": "4.0",
            "species": "Iris-setosa",
        },
        Purpose.TRAINING,
    )
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.purpose == Purpose.TRAINING
    assert sample.species == "Iris-setosa"


def test_testing_known_sample_from_dict() -> None:
    """Test the creation of a known sample from a dict."""
    sample = KnownSample.from_dict(
        {
            "sepal_length": "1.0",
            "sepal_width": "2.0",
            "petal_length": "3.0",
            "petal_width": "4.0",
            "species": "Iris-setosa",
        },
        Purpose.TESTING,
    )
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.purpose == Purpose.TESTING
    assert sample.species == "Iris-setosa"


def test_testing_known_sample_from_dict_raises_error_on_wrong_species() -> None:
    """Test the creation of a known sample from a dict."""
    with pytest.raises(InvalidSampleError) as excinfo:
        KnownSample.from_dict(
            {
                "sepal_length": "1.0",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "4.0",
                "species": "Iris-unknown",
            },
            Purpose.TESTING,
        )

    assert str(excinfo.value).startswith("Invalid sample in row:")


def test_testing_known_sample_from_dict_out_of_bounds() -> None:
    """Test the creation of a known sample from a dict."""
    with pytest.raises(OutOfBoundsError):
        KnownSample.from_dict(
            {
                "sepal_length": "1.0",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "-3.0",
                "species": "Iris-setosa",
            },
            Purpose.TESTING,
        )


def test_testing_known_sample_from_dict_raises_error_on_invalid_floats() -> None:
    """Test the creation of a known sample from a dict."""
    with pytest.raises(InvalidSampleError) as excinfo:
        KnownSample.from_dict(
            {
                "sepal_length": "unknown",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "4.0",
                "species": "Iris-setosa",
            },
            Purpose.TESTING,
        )

    assert str(excinfo.value).startswith("Invalid sample in row:")


def test_testing_known_sample_from_dict_raises_error_on_missing_values() -> None:
    """Test the creation of a known sample from a dict."""
    with pytest.raises(InvalidSampleError) as excinfo:
        KnownSample.from_dict(
            {
                "sepal_length": "1.0",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "4.0",
            },
            Purpose.TESTING,
        )

    assert str(excinfo.value).startswith("Missing column in row:")


def test_matches_raises_error_on_training_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TRAINING, "Iris-setosa")
    with pytest.raises(AttributeError):
        sample.matches()


def test_matches_of_correctly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING, "Iris-setosa")
    sample.classification = "Iris-setosa"
    assert sample.matches()


def test_matches_of_uncorrectly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, Purpose.TESTING, "Iris-setosa")
    sample.classification = "Iris-versicolor"
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
        "petal_width=4.0)"
    )
