from treeNode import TreeNode
import numpy as np
import pandas as pd

class TreeC4_5:
  def __init__(self,X,y):
    # initiate the root node when instantiating an object of this class
    self.root = TreeNode(X,y, node_name="Root node")

  def fit(self):
    # fit the tree model
    self.root.make()

  def predict(self,samples):
    '''
    make the predictions based on the fitted model.

    Parameters:
      samples (DataFrame type): The testing datasets

    return value: the predictions array
    '''
    # create a prediction list to store the prediction
    prediction_list = []

    # make the prediction for each sample one by one in the testing datasets
    for sample_index in range(len(samples)):
      # use .iloc[index] to get the sample by the index
      sample = samples.iloc[sample_index]

      # make the prediction for the sample
      prediction = self.make_prediction(sample)

      # append the prediction to the prediction list
      prediction_list.append(prediction)
    return np.array(prediction_list)

  def make_prediction(self,sample):
    '''
    make a prediction for one sample

    Parameters:
      sample(Series type): one row of the testing datasets 

    return value: prediciton result
    '''
    current_node = self.root
    while current_node.decision is None:

      # get the attribute to split in the current node
      attribute_to_split = current_node.split_attr

      # get the value for this attribute in sample 
      attribute_value = sample[attribute_to_split]
      
      # choose the next attribute that the node should move based on the value is higher or lower than the split point
      if attribute_value > current_node.split_point:
        next_attr = current_node.split_attr + ' > ' + str(current_node.split_point)
      
      else:
        next_attr = current_node.split_attr + ' <= ' + str(current_node.split_point)
          
      current_node = current_node.children[next_attr]

    # when decision found in the node, return this decision as the prediction
    return current_node.decision