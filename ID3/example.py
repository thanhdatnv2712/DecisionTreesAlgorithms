# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np

data = pd.read_csv(
        "dataset/weather.csv",
    )
data = data.sample(frac=1)
total_data = len(data)
x_train = np.array((data.iloc[:-50,1:-1].values), dtype='float32')
x_test = np.array((data.iloc[-50:,1:-1].values), dtype='float32')
y_train = np.unique(data.iloc[:-50,-1].values, return_inverse=True)[1]
y_test = np.unique(data.iloc[-50:,-1].values, return_inverse=True)[1]

clf = tree.DecisionTreeClassifier(criterion="gini") 
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))