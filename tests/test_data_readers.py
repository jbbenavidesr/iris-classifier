from pathlib import Path

from iris_classifier.data_readers import SampleReader


def test_sample_reader_init() -> None:
    """Test the initialization of a sample reader."""
    mock_data = Path("tests/mock_data/bezdekIris.data")
    sample_reader = SampleReader(mock_data)
    assert sample_reader.source == mock_data


def test_sample_reader_sample_iter() -> None:
    """Test the sample iterator of a sample reader."""
    mock_data = Path("tests/mock_data/bezdekIris.data")
    sample_reader = SampleReader(mock_data)
    samples = list(sample_reader.sample_iter())
    assert len(samples) == 150
