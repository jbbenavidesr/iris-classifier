"""Different Implementations of pluggable distance functions for the KNN."""

import abc

from .samples import Sample


class Distance(abc.ABC):
    """Base class for a pluggable distance algorithm to use as hyperparameter"""

    @abc.abstractmethod
    def distance(self, a: Sample, b: Sample) -> float:
        """Compute the distance between two samples.

        :param a: The first sample.
        :param b: The second sample.
        :return: The distance between the samples.
        """


class MinkowskiDistance(Distance):
    """Minkowski distance.

    This is a generalization of the Euclidean and Manhattan distances.
    """

    p: int

    def distance(self, a: Sample, b: Sample) -> float:
        return sum(
            [
                abs(a.sepal_length - b.sepal_length) ** self.p,
                abs(a.sepal_width - b.sepal_width) ** self.p,
                abs(a.petal_length - b.petal_length) ** self.p,
                abs(a.petal_width - b.petal_width) ** self.p,
            ]
        ) ** (1 / self.p)


class EuclidianDistance(MinkowskiDistance):
    """Euclidean distance."""

    p = 2


class ManhattanDistance(MinkowskiDistance):
    """Manhattan distance."""

    p = 1


class ChebyshevDistance(Distance):
    """Chebyshev distance."""

    def distance(self, a: Sample, b: Sample) -> float:
        return max(
            [
                abs(a.sepal_length - b.sepal_length),
                abs(a.sepal_width - b.sepal_width),
                abs(a.petal_length - b.petal_length),
                abs(a.petal_width - b.petal_width),
            ]
        )


class SorensenDistance(Distance):
    """Sorensen distance."""

    def distance(self, a: Sample, b: Sample) -> float:
        return sum(
            [
                abs(a.sepal_length - b.sepal_length),
                abs(a.sepal_width - b.sepal_width),
                abs(a.petal_length - b.petal_length),
                abs(a.petal_width - b.petal_width),
            ]
        ) / sum(
            [
                abs(a.sepal_length + b.sepal_length),
                abs(a.sepal_width + b.sepal_width),
                abs(a.petal_length + b.petal_length),
                abs(a.petal_width + b.petal_width),
            ]
        )
