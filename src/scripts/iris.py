from sklearn import datasets
import pickle

IRIS_FILENAME = "src/dataset/iris.dataset.pickle"

try:
    with open(IRIS_FILENAME, "rb") as f:
        dataset = pickle.load(f)
except:
    dataset = datasets.load_iris()
    with open(IRIS_FILENAME, "wb") as f:
        pickle.dump(dataset, f) 
