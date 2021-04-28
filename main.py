import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data)
    X.columns=['Sepal_length', 'Sepal_width', 'Petal_length', 'Petal_width']
    y=pd.DataFrame(iris.target)
    y.columns=['Targets']
    print(X)
    print(y)

    model = KMeans(n_clusters=3)
    model.fit(X)
    print(model.labels_)
    plt.scatter(X.Petal_length, X.Petal_width)
    colormap = np.array(['Red', 'green', 'blue'])
    #Affiche fleurs selon leurs classes
    plt.scatter(X.Petal_length, X.Petal_width, c=colormap[y.Targets], s=40)
    plt.show()
    plt.scatter(X.Petal_length, X.Petal_width, c=colormap[model.labels_], s=40)
    plt.show()