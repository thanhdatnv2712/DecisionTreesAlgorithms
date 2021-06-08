import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from treeC4_5 import TreeC4_5

df = pd.read_csv("./dataset/iris.csv")
df.head()

def change_target_type(x):
  '''
    convert the string type of the target values to numerical type
    in order to visualize the results in the scatter plot later
  '''
  if x == 'Iris-setosa':
    return 0

  elif x == 'Iris-versicolor':
    return 1
  
  elif x == 'Iris-virginica':
    return 2

df.loc[:,"class"] = df["class"].apply(change_target_type)
df.head()

attrs = df.keys()[:-1]

# split the datasets into the attributes datasets X_total (DataFrame type) and target datasets y_total (Series type)
print("The number of samples in the total datasets are {}\n".format(df.shape[0]))
X_total = df[attrs]
y_total = df.iloc[:,-1]

# Set the random_state = 48 (random seed) to guarantee that my split will be always the same
X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size = 0.33,random_state = 48)


print("The number of samples in the training datasets are {}".format(X_train.shape[0]))
print("The number of samples in the testing datasets are {}".format(X_test.shape[0]))

dtc4_5 = TreeC4_5(X_train,y_train)
dtc4_5.fit()

decisions = dtc4_5.predict(X_test)

# got the boolean result that whther the prediction is true of false
results = decisions == y_test

# calculate the accuracy and output
accuracy = sum(results) / len(results)
print("My model's accuracy for the iris dataset is: ", accuracy)
