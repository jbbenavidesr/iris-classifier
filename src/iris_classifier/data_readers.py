"""Some classes for reading data from different sources."""
import csv
import json
from pathlib import Path
from typing import Any, Iterator, TypedDict

import jsonschema

from .exceptions import BadSampleRow
from .models import KnownSample, Sample


class SampleDict(TypedDict):
    """Dictionary read from input data"""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str


class SampleReader:
    """Reads samples from a CSV file."""

    header = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

    def __init__(self, source: Path) -> None:
        self.source = source

    def sample_iter(self) -> Iterator[KnownSample]:
        """Returns an iterator over the samples in the source file."""
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            for row in reader:
                try:
                    sample = KnownSample(
                        sample=Sample(
                            sepal_length=float(row["sepal_length"]),
                            sepal_width=float(row["sepal_width"]),
                            petal_length=float(row["petal_length"]),
                            petal_width=float(row["petal_width"]),
                        ),
                        species=row["class"],
                    )
                except ValueError as ex:
                    raise BadSampleRow(f"Invalid sample row: {row!r}") from ex

                yield sample


class CSVIrisReader:
    """
    Attribute Information:
        1. sepal length in cm
        2. sepal width in cm
        3. petal length in cm
        4. petal width in cm
        5. class:
            -- Iris Setosa
            -- Iris Versicolour
            -- Iris Virginica
    """

    header = [
        "sepal_length",  # in cm
        "sepal_width",  # in cm
        "petal_length",  # in cm
        "petal_width",  # in cm
        "species",  # Iris Setosa, Iris Versicolour, Iris Virginica
    ]

    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            reader = csv.DictReader(source_file, self.header)
            yield from reader


class JSONIrisReader:
    """Reads Iris data from a JSON file."""

    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            sample_list = json.load(source_file)

        yield from sample_list


class NDJSONIrisReader:
    """Reads Iris data from a JSON file."""

    def __init__(self, source: Path) -> None:
        self.source = source

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            for line in source_file:
                sample = json.loads(line)
                yield sample


class ValidatingNDJSONIrisReader:
    """Reads Iris data from an NDJSON file and validates the data using a schema."""

    def __init__(self, source: Path, schema: dict[str, Any]) -> None:
        self.source = source
        self.validator = jsonschema.Draft7Validator(schema)

    def data_iter(self) -> Iterator[SampleDict]:
        with self.source.open() as source_file:
            for line in source_file:
                sample = json.loads(line)
                if self.validator.is_valid(sample):
                    yield sample
                else:
                    print(f"Invalid sample: {sample!r}")
