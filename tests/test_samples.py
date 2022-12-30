from iris_classifier.models import (
    KnownSample,
    Sample,
    TestingKnownSample,
    TrainingKnownSample,
    UnknownSample,
)


def test_sample_init() -> None:
    """Test the initialization of a sample."""
    sample = Sample(1.0, 2.0, 3.0, 4.0)
    assert sample.sepal_length == 1.0
    assert sample.sepal_width == 2.0
    assert sample.petal_length == 3.0
    assert sample.petal_width == 4.0


def test_unknown_sample_init(sample: Sample) -> None:
    """Test the initialization of an unknown and unclassified sample."""
    unknown_sample = UnknownSample(sample)
    assert unknown_sample.sample == sample


def test_known_sample_init(sample: Sample) -> None:
    """Test the initialization of a known sample."""
    known_sample = KnownSample(sample, "Iris-setosa")
    assert known_sample.sample == sample
    assert known_sample.species == "Iris-setosa"


def test_training_known_sample_init(known_sample: KnownSample) -> None:
    """Test the initialization of a training known sample."""
    training_sample: TrainingKnownSample = TrainingKnownSample(known_sample)
    assert training_sample.sample == known_sample


def test_testing_known_sample_init(known_sample: KnownSample) -> None:
    """Test the initialization of a testing known sample."""
    testing_sample: TestingKnownSample = TestingKnownSample(known_sample)
    assert testing_sample.sample == known_sample
    assert testing_sample.classification is None


def test_testing_known_sample_matches(known_sample: KnownSample) -> None:
    """Test the matches method of a testing known sample."""
    sample = TestingKnownSample(known_sample)
    sample.classification = known_sample.species
    assert sample.matches() is True


def test_testing_known_sample_does_not_match(known_sample: KnownSample) -> None:
    """Test the matches method of a testing known sample."""
    sample = TestingKnownSample(known_sample)
    sample.classification = "Iris-versicolor"
    assert sample.matches() is False
