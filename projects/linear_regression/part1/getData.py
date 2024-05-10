import pandas as pd
import numpy as np
from typing import Tuple


def getDataSplits() -> pd.DataFrame:
    """
    Returns the train and test split.

    Args:
        None

    Returns:
        train: train set shuffled * 0.8
        test: test set shuffled * 0.2
    """
    # data folder path
    data_folder = "../data/"
    data_file = "admission.csv"

    # Set seed for reproducibility
    np.random.seed(seed=1234)

    # make a clean copy and shuffle it
    data = pd.read_csv(data_folder + data_file)
    clean = data.copy()
    clean = clean.sample(frac=1)

    clean.rename(
        columns={"LOR ": "LOR", "Chance of Admit ": "Chance of Admit"}, inplace=True
    )

    # drop the id of student
    clean.drop("Serial No.", axis=1, inplace=True)

    # Make 2 split, train and test
    size_train = int(len(clean) * 0.8)
    size_test = len(clean) - size_train

    train = clean.iloc[:size_train]
    test = clean.iloc[size_train : size_train + size_test]
    print("done")

    return train, test


def transformData(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform the data.

    Args:
        train: the train set
        test: the test set

    Returns:
        train: the train set transformed
        test: the test set transformed
    """

    x_train = train.copy()
    y_train = train["Chance of Admit"]
    x_train = x_train.drop(columns=["Chance of Admit"])

    x_test = test.copy()
    y_test = test["Chance of Admit"]
    x_test = x_test.drop(columns=["Chance of Admit"])

    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()

    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    return x_train, y_train, x_test, y_test


def main():
    train, test = getDataSplits()


if __name__ == "__main__":
    main()
