import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_mldata
dataset = fetch_mldata('MNIST original')

from sklearn.datasets import load_digits
dataset_1 = load_digits()


X = dataset_1.data
y = dataset_1.target

some_digit = X[1328]
some_digit_image = some_digit.reshape(8,8)

plt.imshow(some_digit_image)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = train_test_split(X,y)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth = 15)
dtc.fit(X_train , y_train)
dtc.score(X_test , y_test)

dtc.predict(X[[369,123,412,1328] , 0:64])

from sklearn.tree import export_graphviz

export_graphviz(dtc, out_file="tree.dot")


import graphviz
with open ("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
