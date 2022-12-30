"""Some classes for reading data from different sources."""
import csv
from pathlib import Path
from typing import Iterator

from .exceptions import BadSampleRow
from .models import KnownSample, Sample


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
