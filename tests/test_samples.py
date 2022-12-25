from iris_classifier.samples import Sample


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
