import numpy as np
import pandas as pd

class Myutils:
  def get_split_pointSet(self,attr_series):
    '''
    Discretize continuous attributes through the dichotomy

    Parameters:
      attr_series (Series type) : The value of an attribute column passed in 

    return value：The array with all division points
    '''

    # use the .unique() functioon to get the unique values from the input
    # the .unique() function can automatically sort the array when recalling
    unique_values = np.unique(attr_series)

    # use the list comprehension to calculate the arithmetic mean between two points in unique_values
    result = [(unique_values[i] + unique_values[i+1]) / 2  for i in range(len(unique_values)-1)]

    return np.unique(result)
    
  def get_discrete_variables(self,X,attr,split_point):
    '''
    Convert continuous variables through the split point into categorical variables 
    which greater than the split point or smaller than the split point

    Parameters:
      X (DataFrame type) : The passed in datasets 
      attr (String type) : used to select the related attribute column 
      split_point (float type) : Divided/split point 

    return value: the new column with categorical variables 
    '''

    # make the sequence of scalars in bins_to_cut list
    bins_to_cut = [min(X[attr]),split_point,max(X[attr])]

    # Specifies the labels for the returned bins in labels_to_cut list
    labels_to_cut = [attr + ' < ' + str(split_point), attr + ' > ' + str(split_point)]

    # use .cut() function to get the categorical variables column, left-include the first interval
    categorized_column = pd.cut(X[attr], bins= bins_to_cut , labels = labels_to_cut, include_lowest=True)
    return categorized_column

  def compute_entropy(self,y):
    '''
    calculate the information entropy 

    Parameters:
      y (Series type): the target values passed in 

    return value: 0.0 or calculated information entropy
    '''
    
    # if the length of y less than 2, it means there are no more uncertainty
    # in this case, the entropy is 0.0
    if len(y) < 2:
      return 0.0
    
    # if the unique values in y are less than 2, it means that all sampls have the same feature values
    # in this case, the entropy is 0.0
    if len(np.unique(y)) < 2:
      return 0.0

    # calculate the counts of value in y series, the normalize =True means the counts are presented as the frequency
    freq = y.value_counts(normalize=True)

    # To check whether there is a frequency that is 0.0 before the log calculation
    if (freq == 0).any():
      # Use the boolean mask to filter the frequency that is 0.0
      freq = freq[freq != 0]
      return -(freq * np.log2(freq)).sum()

    else:
      return -(freq * np.log2(freq)).sum()

  def compute_info_gain_ratio(self,X,y,attr,split_point_list):
    '''
    Calculate the information gain ratio of the specified attribute for a continuous variable
    
    Parameters:
      X (DataFrame type): Datasets with the attributes values
      y (Series type): Datasets with the target values 
      attr (String type): The attribute for calculating the information gain rate 
      split_point_lst (numpy.array type): Divided points array

    return values: Result of the calculate information gain ratio, and the best split point of float type     
    '''
    # Define the variables best_split_point, best_info_gain, split_entropy, best_categorized_column
    best_split_point = 0.0
    best_info_gain = 0.0
    split_entropy = 0.0
    best_categorized_column = None 

    # if the length of the split_point_list is 0, it means there are no split point
    # further more, there is no need to consider this attribute
    # so the result could be 0.0 and None
    if len(split_point_list) == 0:
      return 0.0, None

    else:
      # use the for loop to calculate the best information gain and best split point
      for split_point in split_point_list:

        # Divide the X datasets to the sub-datasets that the values are lower than the split point 
        lower_index_boolean = X[attr] < split_point
        lower_X = X[lower_index_boolean]
        lower_y = y[lower_index_boolean]

        # Divide the X datasets to the sub-datasets that the values are higher than the split point
        upper_index_boolean = X[attr] > split_point
        upper_X = X[upper_index_boolean]
        upper_y = y[upper_index_boolean]
        
        # compute the entropy of target values
        Ent_D = self.compute_entropy(y)
        
        # Calculate the probability of these two sub-data sets occupying the total data set separately
        lower_prob = len(lower_X) / len(X)
        upper_prob = len(upper_X) / len(X)

        # compute the entropy of the sub-datasets that all the values are lower than the split point
        lowersets_entropy = lower_prob * self.compute_entropy(lower_y)

        # compute the entropy of the sub-datasets that all the values are higher than the split point
        uppersets_entropy = upper_prob * self.compute_entropy(upper_y)   
        
        # compute the information gain for this split point
        info_gain = Ent_D - (lowersets_entropy + uppersets_entropy)
        # get the categorical values for these numerical values lower or higher than the split point 
        categorized_column = self.get_discrete_variables(X,attr,split_point)
        # select the best information gain and best split point
        if info_gain > best_info_gain:
          best_info_gain = info_gain
          best_split_point = split_point
          best_categorized_column = categorized_column

      # If there is only one split point in the split point list and
      # the informaiton gain of this one split point is zero, then the best split point is also 0
      if best_info_gain == 0:
        return 0.0, best_split_point

      # computer the split information entropy
      split_entropy = self.compute_entropy(best_categorized_column)

      # use the best information gain to compute the information gain ratio
      # if the split information entroy is zero, it means this attribute is constant
      if split_entropy == 0:
        return "undefined", None
        
      else:
        info_gain_ratio = best_info_gain / split_entropy
        return info_gain_ratio , best_split_point