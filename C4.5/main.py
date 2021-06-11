import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from treeC4_5 import TreeC4_5

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--dataset', default='wine', help='dataset name')
parser.add_argument('--is_discrete', default=0, type=int, help='flag discrete dataset')
parser.add_argument('--vis', default=0, type=int, help='visualization')
args = parser.parse_args()

def change_target_type_iris(x):
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

def change_target_type_wine(x):
  if x > 6.5:
    return 'good'

  else:
    return 'not good'

def change_target_type_weather(x):
  if x == 'no':
    return 0

  elif x == 'sunny' or x == 'hot' or x == 'high' or x == 'weak' or x == 'yes':
    return 1

  elif x == 'rainy' or x == 'cool' or x == 'normal' or x == 'strong':
    return 2

  elif x == 'overcast' or x == 'mild':
    return 3
  
if __name__ == '__main__':
  dataset_path = None
  is_discrete = args.is_discrete
  change_target_type = None
  class_apply = None
  if args.dataset == "iris":
    dataset_path = "../dataset/iris.csv"
    change_target_type = change_target_type_iris
    class_apply = "class"
  elif args.dataset == "trans":
    dataset_path = "../dataset/transfusion.csv"
  elif args.dataset == "car":
    dataset_path = "../dataset/car.csv"
  elif args.dataset == "weather":
    dataset_path = "../dataset/weather.csv"
    change_target_type = change_target_type_weather
  elif args.dataset == "wine":
    dataset_path = "../dataset/winequality-red.csv"
    change_target_type = change_target_type_wine
    class_apply = "quality"
  else:
    print ("[ERROR] DO NOT SUPPORT DATASET!")
    exit(0)

  # df = pd.read_csv(dataset_path)
  # df.head()
  # if change_target_type is not None:
  #   df.loc[:, class_apply] = df[class_apply].apply(change_target_type)
  #   df.head()
  # attrs = df.keys()[:-1]

  # print("The number of samples in the total datasets are {}\n".format(df.shape[0]))
  # X_total = df[attrs]
  # y_total = df.iloc[:,-1]
  # # X_total.to_csv("../dataset/winequality-red_new.csv", index=False)
  # X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size = 0.33,random_state = 49)


  # df_train = pd.read_csv("../dataset/iris_train_20.csv")
  # df_train.head()
  # if change_target_type is not None:
  #   df_train.loc[:, class_apply] = df_train[class_apply].apply(change_target_type)
  #   df_train.head()
  # attrs = df_train.keys()[:-1]

  # X_train = df_train[attrs]
  # y_train = df_train.iloc[:,-1]

  # df_test = pd.read_csv("../dataset/iris_test_20.csv")
  # df_test.head()
  # if change_target_type is not None:
  #   df_test.loc[:, class_apply] = df_test[class_apply].apply(change_target_type)
  #   df_test.head()
  # attrs = df_test.keys()[:-1]

  # X_test = df_test[attrs]
  # y_test = df_test.iloc[:,-1]

  # print("The number of samples in the training datasets are {}".format(X_train.shape[0]))
  # print("The number of samples in the testing datasets are {}".format(X_test.shape[0]))

  # dtc4_5 = TreeC4_5(X_train,y_train)
  # dtc4_5.fit()

  # decisions = dtc4_5.predict(X_test)

  # results = decisions == y_test

  # accuracy = sum(results) / len(results)
  # print("My model's accuracy for the {} dataset is: ".format(args.dataset), accuracy, "\n")
  # if args.vis:
  #   vis = {}
  #   dtc4_5.root.print_node_details(vis)
  #   dtc4_5.root.print_tree(vis)

  df_train = pd.read_csv("../dataset/winequality-red_train_33.csv")
  df_train.head()
  attrs = df_train.keys()[:-1]

  X_train = df_train[attrs]
  y_train = df_train.iloc[:,-1]

  df_test = pd.read_csv("../dataset/winequality-red_test_33.csv")
  df_test.head()
  attrs = df_test.keys()[:-1]

  X_test = df_test[attrs]
  y_test = df_test.iloc[:,-1]

  print("The number of samples in the training datasets are {}".format(X_train.shape[0]))
  print("The number of samples in the testing datasets are {}".format(X_test.shape[0]))

  dtc4_5 = TreeC4_5(X_train,y_train)
  dtc4_5.fit()

  decisions = dtc4_5.predict(X_test)

  results = decisions == y_test

  accuracy = sum(results) / len(results)
  print("My model's accuracy for the {} dataset is: ".format(args.dataset), accuracy, "\n")
  if args.vis:
    vis = {}
    dtc4_5.root.print_node_details(vis)
    dtc4_5.root.print_tree(vis)
