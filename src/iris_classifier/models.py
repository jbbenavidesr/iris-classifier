"""Simple application for classifying iris flowers.

This application will have 2 users: A botanist, which is the one who will add the
classified data of iris flowers that will be used to train the model. And a user which
wants an unknown sample of iris flower to be classified by the application. The botanist
will select the hyperparameters used by the model.
"""
import datetime
import weakref
from typing import Iterable, Optional


class Sample:
    """A sample of an iris flower."""

    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        spicies: Optional[str] = None,
    ) -> None:
        """Initialize a sample of an iris flower.

        :param sepal_length: The length of the sepal.
        :param sepal_width: The width of the sepal.
        :param petal_length: The length of the petal.
        :param petal_width: The width of the petal.
        :param spicies: The spicies of the flower. If the sample is unknown, this value
            should be None.
        """
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.spicies = spicies
        self.classification: Optional[str] = None

    def __repr__(self) -> str:
        if self.spicies is None:
            known_unknown = "UnknownSample"
        else:
            known_unknown = "KnownSample"

        if self.classification is None:
            classification = ""
        else:
            classification = f", classification={self.classification}"

        return (
            f"{known_unknown}("
            f"sepal_length={self.sepal_length}, "
            f"sepal_width={self.sepal_width}, "
            f"petal_length={self.petal_length}, "
            f"petal_width={self.petal_width}, "
            f"spicies={self.spicies!r}{classification})"
        )

    def classify(self, classification: str) -> None:
        """Classify the sample.

        :param classification: The classification of the sample.
        """
        self.classification = classification

    def matches(self) -> bool:
        """Check if the classification matches the spicies.

        :return: True if the classification matches the spicies, False otherwise.
        """
        return self.spicies == self.classification


class Hyperparameter:
    """A hyperparameter set of values for the model and the overall quality of the
    classification."""

    def __init__(self, k: int, training: "TrainingData") -> None:
        """Initialize a hyperparameter set.

        :param k: The k value for the model.
        :param training: The training data used to train the model.
        """
        self.k = k
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        """Test the hyperparameter set."""
        training_data: Optional["TrainingData"] = self.data()
        if not training_data:
            raise RuntimeError(
                "Training data is not available anymore. Broken Weak Reference."
            )

        pass_count, fail_count = 0, 0
        for sample in training_data.testing:
            sample.classify(self.classify(sample))
            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1

        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Sample) -> str:
        """Classify a sample.

        :param sample: The sample to classify.
        :return: The classification of the sample.
        """
        # For testing
        return "Iris-setosa"


class TrainingData:
    """Set of training  and testing data used to train the model. It has methods to load
    and test the samples."""

    def __init__(self, name: str) -> None:
        """Initialize the training data.

        :param name: The name of the training data.
        """
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[Sample] = []
        self.testing: list[Sample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_source: Iterable[dict[str, str]]) -> None:
        """Load the raw data source and partition it into training and testing data."""
        for n, raw_sample in enumerate(raw_data_source):
            sample = Sample(
                float(raw_sample["sepal_length"]),
                float(raw_sample["sepal_width"]),
                float(raw_sample["petal_length"]),
                float(raw_sample["petal_width"]),
                raw_sample["species"],
            )
            if n % 10 == 0:
                self.testing.append(sample)
            else:
                self.training.append(sample)
        self.uploaded = datetime.datetime.now(tz=datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:
        """Test the hyperparameter set.

        :param parameter: The hyperparameter set to test.
        """
        parameter.test()
        self.tuning.append(parameter)
        self.tested = datetime.datetime.now(tz=datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: Sample) -> Sample:
        """Classify a sample.

        :param sample: The sample to classify.
        :param parameter: The hyperparameter set to use.
        :return: The classified sample.
        """
        classification = parameter.classify(sample)
        sample.classify(classification)
        return sample
