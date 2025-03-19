from sklearn import datasets
import pickle
import numpy as np

from sklearn.model_selection import train_test_split


IRIS_FILENAME = "iris.dataset.pickle"

try:
    with open(IRIS_FILENAME, "rb") as f:
        dataset = pickle.load(f)
except:
    dataset = datasets.load_iris()
    with open(IRIS_FILENAME, "wb") as f:
        pickle.dump(dataset, f)
        
X = dataset.data  # type: ignore
y = dataset.target  # type: ignore

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=60 / 150, stratify=y, random_state=99
)