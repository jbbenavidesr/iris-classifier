from pathlib import Path

from iris_classifier.data_readers import CSVIrisReader, SampleReader


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


def test_csv_iris_reader_init() -> None:
    """Test the initialization of a CSV Iris reader."""
    test_data = Path.cwd() / "data" / "bezdekIris.data"
    print(test_data)
    csv_iris_reader = CSVIrisReader(test_data)
    assert csv_iris_reader.source == test_data


def test_csv_iris_reader_reads_150_samples() -> None:
    """Test that the CSV Iris reader reads 150 samples."""
    test_data = Path.cwd() / "data" / "bezdekIris.data"
    csv_iris_reader = CSVIrisReader(test_data)
    samples = list(csv_iris_reader.data_iter())
    assert len(samples) == 150


def test_csv_iris_reader_item_is_iris_dict() -> None:
    """Test that the CSV Iris reader reads 150 samples."""
    test_data = Path.cwd() / "data" / "bezdekIris.data"
    csv_iris_reader = CSVIrisReader(test_data)
    samples = list(csv_iris_reader.data_iter())
    assert len(samples) == 150
    assert isinstance(samples[0], dict)
    assert samples[0]["sepal_length"] == "5.1"
    assert samples[0]["sepal_width"] == "3.5"
    assert samples[0]["petal_length"] == "1.4"
    assert samples[0]["petal_width"] == "0.2"
    assert samples[0]["species"] == "Iris-setosa"
