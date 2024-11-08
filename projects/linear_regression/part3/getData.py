import pandas as pd
import numpy as np
import torch
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def getDataSplits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the train and test split.

    Args:
        None

    Returns:
        train: train set shuffled * 0.8
        test: test set shuffled * 0.2
    """
    # data folder path
    data_path = "../data/admission.csv"

    # Make a clean copy and shuffle it
    data = pd.read_csv(data_path)

    # Rename columns to standardize names
    data.rename(
        columns={"LOR ": "LOR", "Chance of Admit ": "Chance of Admit"}, inplace=True
    )

    # Drop the student ID
    data.drop("Serial No.", axis=1, inplace=True)

    # Split data into 80% train, 20% test
    train, test = train_test_split(data, test_size=0.2, random_state=1234)

    return train, test


def transformData(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transforms the data into PyTorch tensors.

    Args:
        train: the train set as DataFrame
        test: the test set as DataFrame

    Returns:
        x_train: features of train set as tensor
        y_train: labels of train set as tensor
        x_test: features of test set as tensor
        y_test: labels of test set as tensor
    """
    # Separate features and target variable for training data
    x_train = train.drop(columns=["Chance of Admit"]).values
    y_train = train["Chance of Admit"].values

    # Separate features and target variable for testing data
    x_test = test.drop(columns=["Chance of Admit"]).values
    y_test = test["Chance of Admit"].values

    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


    return x_train, y_train, x_test, y_test


def main():
    train, test = getDataSplits()
    x_train, y_train, x_test, y_test = transformData(train, test)
    print("Data loaded and transformed into PyTorch tensors.")

if __name__ == "__main__":
    main()
