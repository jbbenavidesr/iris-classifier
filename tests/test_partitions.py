from iris_classifier.partitions import (
    partition_samples,
    training_67,
    training_75,
    training_80,
    training_90,
)
from iris_classifier.samples import KnownSample


def test_training_67_rule(known_sample: KnownSample):
    """Test the training_67 rule."""
    assert not training_67(known_sample, 0)
    assert training_67(known_sample, 1)
    assert training_67(known_sample, 2)
    assert not training_67(known_sample, 3)
    assert training_67(known_sample, 4)
    assert training_67(known_sample, 5)
    assert not training_67(known_sample, 6)
    assert training_67(known_sample, 7)
    assert training_67(known_sample, 8)
    assert not training_67(known_sample, 9)
    assert training_67(known_sample, 10)
    assert training_67(known_sample, 11)
    assert not training_67(known_sample, 12)


def test_training_75_rule(known_sample: KnownSample):
    """Test the training_75 rule."""
    assert not training_75(known_sample, 0)
    assert training_75(known_sample, 1)
    assert training_75(known_sample, 2)
    assert training_75(known_sample, 3)
    assert not training_75(known_sample, 4)
    assert training_75(known_sample, 5)
    assert training_75(known_sample, 6)
    assert training_75(known_sample, 7)
    assert not training_75(known_sample, 8)
    assert training_75(known_sample, 9)
    assert training_75(known_sample, 10)
    assert training_75(known_sample, 11)
    assert not training_75(known_sample, 12)


def test_training_80_rule(known_sample: KnownSample):
    """Test the training_80 rule."""
    assert not training_80(known_sample, 0)
    assert training_80(known_sample, 1)
    assert training_80(known_sample, 2)
    assert training_80(known_sample, 3)
    assert training_80(known_sample, 4)
    assert not training_80(known_sample, 5)
    assert training_80(known_sample, 6)
    assert training_80(known_sample, 7)
    assert training_80(known_sample, 8)
    assert training_80(known_sample, 9)
    assert not training_80(known_sample, 10)


def test_training_90_rule(known_sample: KnownSample):
    """Test the training_90 rule."""
    assert not training_90(known_sample, 0)
    assert training_90(known_sample, 1)
    assert training_90(known_sample, 2)
    assert training_90(known_sample, 3)
    assert training_90(known_sample, 4)
    assert training_90(known_sample, 5)
    assert training_90(known_sample, 6)
    assert training_90(known_sample, 7)
    assert training_90(known_sample, 8)
    assert training_90(known_sample, 9)
    assert not training_90(known_sample, 10)


def test_partition_samples(mock_training_data: list[KnownSample]):
    """Test partition_samples."""
    samples = mock_training_data
    training_samples, testing_samples = partition_samples(samples, training_67)
    assert len(training_samples) == 20
    assert len(testing_samples) == 10
    training_samples, testing_samples = partition_samples(samples, training_75)
    assert len(training_samples) == 22
    assert len(testing_samples) == 8
    training_samples, testing_samples = partition_samples(samples, training_80)
    assert len(training_samples) == 24
    assert len(testing_samples) == 6
    training_samples, testing_samples = partition_samples(samples, training_90)
    assert len(training_samples) == 27
    assert len(testing_samples) == 3
