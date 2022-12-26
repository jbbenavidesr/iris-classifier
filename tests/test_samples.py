import pytest

from iris_classifier.exceptions import InvalidSampleError
from iris_classifier.samples import KnownSample, Sample, UnknownSample


def test_sample_init() -> None:
    """Test the initialization of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.classification is None


def test_sample_repr_of_unclassified_sample() -> None:
    """Test the repr of an unknown and unclassified sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    assert repr(sample) == (
        "Sample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0)"
    )


def test_sample_classify() -> None:
    """Test the classification of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    sample.classify("Iris-setosa")
    assert sample.classification == "Iris-setosa"


def test_sample_repr_of_classified_sample() -> None:
    """Test the repr of an unknown and classified sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    sample.classify("Iris-setosa")
    assert repr(sample) == (
        "Sample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "classification=Iris-setosa)"
    )


def test_known_sample_init() -> None:
    """Test the initialization of a known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.species == "Iris-setosa"


def test_known_sample_unclassified_repr() -> None:
    """Test the repr of an unclassified known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert repr(sample) == (
        "KnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species=Iris-setosa)"
    )


def test_known_sample_classified_repr() -> None:
    """Test the repr of a classified known sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    sample.classify("Iris-setosa")
    assert repr(sample) == (
        "KnownSample("
        "sepal_length=1.0, "
        "sepal_width=2.0, "
        "petal_length=3.0, "
        "petal_width=4.0, "
        "species=Iris-setosa, "
        "classification=Iris-setosa)"
    )


def test_known_sample_from_dict() -> None:
    """Test the creation of a known sample from a dict."""
    sample = KnownSample.from_dict(
        {
            "sepal_length": "1.0",
            "sepal_width": "2.0",
            "petal_length": "3.0",
            "petal_width": "4.0",
            "species": "Iris-setosa",
        }
    )
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0
    assert sample.species == "Iris-setosa"


def test_known_sample_from_dict_raises_error_on_wrong_species() -> None:
    """Test the creation of a known sample from a dict."""
    with pytest.raises(InvalidSampleError) as excinfo:
        KnownSample.from_dict(
            {
                "sepal_length": "1.0",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "4.0",
                "species": "Iris-unknown",
            }
        )

    assert str(excinfo.value).startswith("Invalid sample in row:")


def test_known_sample_from_dict_raises_error_on_invalid_floats() -> None:
    """Test the creation of a known sample from a dict."""
    with pytest.raises(InvalidSampleError) as excinfo:
        KnownSample.from_dict(
            {
                "sepal_length": "unknown",
                "sepal_width": "2.0",
                "petal_length": "3.0",
                "petal_width": "4.0",
                "species": "Iris-setosa",
            }
        )

    assert str(excinfo.value).startswith("Invalid sample in row:")


def test_matches_of_unclassified_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    assert not sample.matches()


def test_matches_of_correctly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    sample.classify("Iris-setosa")
    assert sample.matches()


def test_matches_of_uncorrectly_classified_sample() -> None:
    """Test the matching of a sample."""
    sample = KnownSample(1.0, 2.0, 3.0, 4.0, "Iris-setosa")
    sample.classify("Iris-versicolor")
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
