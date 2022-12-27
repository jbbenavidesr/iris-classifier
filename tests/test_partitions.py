from iris_classifier.partitions import ShufflingSamplePartition

from .mock_data import training_data as mock_training_data


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
