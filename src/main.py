"""Entry point for the application"""
import time

from iris_classifier.models import (
    TrainingKnownSample,
    TestingKnownSample,
    Hyperparameter,
    TrainingList,
    TestingList,
)
from iris_classifier.classifiers import Classifier, k_nn_1, k_nn_b, k_nn_q
from iris_classifier.distances import manhattan


def a_lot_of_data(n: int) -> tuple[TrainingList, TestingList]:
    """Generate a lot of data for testing"""
    return [], []


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
        f"| {classifier.__name__:10s} | {q:5}/{len(testing_data):5} | {end - start:6.3f}s |"
    )


def main() -> None:
    test, train = a_lot_of_data(5_000)
    print("| algorithm  |  test quality  |  time  |")
    print("|------------|----------------|--------|")
    test_classifier(train, test, k_nn_1)
    test_classifier(train, test, k_nn_b)
    test_classifier(train, test, k_nn_q)


if __name__ == "__main__":
    main()
