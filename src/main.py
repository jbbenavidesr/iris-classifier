"""Entry point for the application"""
import random
import time

from iris_classifier.classifiers import Classifier, k_nn_1, k_nn_b, k_nn_q
from iris_classifier.distances import manhattan
from iris_classifier.models import (
    Hyperparameter,
    KnownSample,
    Sample,
    TestingKnownSample,
    TestingList,
    TrainingKnownSample,
    TrainingList,
)
from iris_classifier.partitions import partition_samples, training_80


def a_lot_of_data(n: int) -> tuple[TrainingList, TestingList]:
    """Generate a lot of random data for testing"""
    samples = (
        KnownSample(
            sample=Sample(
                random.random() * 10,
                random.random() * 10,
                random.random() * 10,
                random.random() * 10,
            ),
            species=random.choice(
                ["Iris-virginica", "Iris-versicolor", "Iris-virginica"]
            ),
        )
        for _ in range(n)
    )

    return partition_samples(samples, training_80)


def test_classifier(
    training_data: list[TrainingKnownSample],
    testing_data: list[TestingKnownSample],
    classifier: Classifier,
) -> None:
    h = Hyperparameter(
        k=3,
        distance_function=manhattan,
        training_data=training_data,
        classifier=classifier,
    )
    start = time.perf_counter()
    q = h.test(testing_data)
    end = time.perf_counter()
    print(
        f"| {classifier.__name__:10s} "
        f"| q={q:5}/{len(testing_data):5} "
        f"| {end - start:6.3f}s |"
    )


def main() -> None:
    train, test = a_lot_of_data(5_000)
    print("| algorithm  |  test quality  |  time  |")
    print("|------------|----------------|--------|")
    test_classifier(train, test, k_nn_1)
    test_classifier(train, test, k_nn_b)
    test_classifier(train, test, k_nn_q)


if __name__ == "__main__":
    main()
