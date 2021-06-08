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

df2 = pd.read_csv("./dataset/transfusion.csv")
df2.head()

attrs2 = df2.keys()[:-1]
print("The number of samples in the total datasets are {}\n".format(df2.shape[0]))
X_total2 = df2[attrs2]
y_total2 = df2.iloc[:,-1]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_total2, y_total2, test_size = 0.33,random_state = 49)

print("The number of samples in the training datasets are {}".format(X_train2.shape[0]))
print("The number of samples in the testing datasets are {}".format(X_test2.shape[0]))

dt2 = TreeC4_5(X_train2,y_train2)
dt2.fit()
decisions = dt2.predict(X_test2)
results = decisions == y_test2
accuracy = sum(results) / len(results)
print("My model's accuracy for the evaluation dataset 1 is: ", accuracy)

df3 = pd.read_csv("./dataset/winequality-red.csv")

# Here I randomly select 750 samples of the total
df3 = df3.sample(n=750, random_state=50, axis=0)

def change_target_type_w(x):
  if x > 6.5:
    return 'good'

  else:
    return 'not good'

df3.loc[:,"quality"] = df3["quality"].apply(change_target_type_w)
df3.head()

attrs3 = df3.keys()[:-1]
print("The number of samples in the total datasets are {}\n".format(df3.shape[0]))
X_total3 = df3[attrs3]
y_total3 = df3.iloc[:,-1]

X_train3, X_test3, y_train3, y_test3 = train_test_split(X_total3, y_total3, test_size = 0.33,random_state = 49)


print("The number of samples in the training datasets are {}".format(X_train3.shape[0]))
print("The number of samples in the testing datasets are {}".format(X_test3.shape[0]))

dt3 = TreeC4_5(X_train3,y_train3)
dt3.fit()
decisions = dt3.predict(X_test3)
results = decisions == y_test3
accuracy = sum(results) / len(results)
print("My model's accuracy for the evaluation dataset 2 is: ", accuracy)
