"""Fixed evaluation samples and the LongCat prompt-length limit."""

import pickle
from pathlib import Path

FEW_SHOT_TEST_SPLIT = 10
MAXIMUM_CONTEXT_LENGTH = 32768


def load_data_split(filename, number_of_few_shots, number_of_tests):
    # Load only the trusted fixtures shipped with this repository.
    path = Path(__file__).resolve().parent / "dataset" / Path(filename).name
    with path.open("rb") as stream:
        data = pickle.load(stream)
    available = len(data) - FEW_SHOT_TEST_SPLIT
    if not 0 <= number_of_few_shots <= FEW_SHOT_TEST_SPLIT:
        raise ValueError(f"Choose 0 to {FEW_SHOT_TEST_SPLIT} few-shot examples.")
    if number_of_tests is None:
        number_of_tests = available
    if not 1 <= number_of_tests <= available:
        raise ValueError(f"Choose 1 to {available} test examples for {path.stem}.")
    return (
        data[:number_of_few_shots],
        data[FEW_SHOT_TEST_SPLIT : FEW_SHOT_TEST_SPLIT + number_of_tests],
    )
