import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = "../dataset/winequality-red_new.csv"
df = pd.read_csv(dataset_path)
df.head()
# if change_target_type is not None:
# df.loc[:, class_apply] = df[class_apply].apply(change_target_type)
# df.head()
attrs = df.keys()[:-1]
all_attrs = df.keys()

print("The number of samples in the total datasets are {}\n".format(df.shape[0]))
X_total = df[all_attrs]
y_total = df.iloc[:,-1]
print (X_total)
print (y_total)
ratio= [0.2, 0.33]
for r in ratio:
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size = r,random_state = 49)
    X_train.to_csv("../dataset/winequality-red_train_{}.csv".format(int(r*100)), index=False)
    X_test.to_csv("../dataset/winequality-red_test_{}.csv".format(int(r*100)), index=False)
    # java -cp /path/to/weka.jar weka.core.converters.CSVLoader filename.csv > filename.arff

