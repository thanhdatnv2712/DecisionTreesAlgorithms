import json
from utils import Myutils

utils = Myutils()

class TreeNode:
  '''
  Construct the entire tree by recursive algorithm
  '''
  def __init__(self,X,y, node_name="", default_decision=None):
    self.X = X # Passed in attributes datasets 
    self.y = y # Passed in target datasets
    self.name = node_name # name of this node
    self.default_decision = default_decision # decision from parent node
    self.decision = None # To record the target value of the leaf node as decision
    self.entropy = 0.0 # record this node's information entropy
    self.attrs_info_gain_ratio = {} # To record the attriubte with its info gain ratio in dictionary structure
    self.split_point = 0.0 # To record the best split point
    self.split_attr = None # To record the best split attribute
    self.split_point_info = [] # store the best split point list, best_split_point, its split attribute and max info gain ratio in this list
    self.children = {} # store the child node in dictionary structure
    # self.vis = {}

  def make(self):
    # use the mode of target datasets to determine the default decision
    # After the root node, the children node should have the default decision passed by their parent node
    if self.default_decision is None:
      self.default_decision = self.y.mode()[0]

    # if there are no data in datasets, use the default decision passed by parent node to determine the decision 
    if len(self.X) == 0:
      self.decision = self.default_decision
      return

    else:
      # use c4.5 algorithm to split the attribute below

      # calculate the information entropy of based on the current target values
      self.entropy = utils.compute_entropy(self.y)
      # if there is only one type of target values, determine the decision by this value 
      target_unique_values = self.y.unique()
      # print (self.y, target_unique_values)
      if len(target_unique_values) == 1:
        self.decision = target_unique_values[0]
        return

      else:
        # initialize the value of max information gain ratio, best split point and best split point list
        max_infoGain_ratio = 0.0
        best_split_point = 0.0
        best_split_point_list = []
        for attr in self.X.keys():
          # get the split point set of this attribute
          split_point_list = utils.get_split_pointSet(self.X[attr])
          if len(split_point_list) == 0:
            # No split point here, which means that this attribute column has only one type of value
            continue

          else:
            # calculate the information gain ratio and splitted point of each attribute
            infoGain_ratio,splited_point = utils.compute_info_gain_ratio(self.X, self.y, attr, split_point_list)
            # store each attribute and its information gain ratio in the dictionary
            self.attrs_info_gain_ratio[attr] = infoGain_ratio

            # select the max information gain ratio
            if infoGain_ratio >= max_infoGain_ratio:
              max_infoGain_ratio = infoGain_ratio
              self.split_attr = attr
              best_split_point_list = split_point_list
              best_split_point = splited_point

        # append the best split point list, best_split_point, its split attribute and max info gain ratio to split_point_info list
        self.split_point_info.append(list(best_split_point_list))
        self.split_point_info.append([best_split_point,self.split_attr,max_infoGain_ratio])
        self.split_point = best_split_point

        # if there exists a split point, the tree could be split into the child node
        if self.split_point != 0.0:
          # make a copy of attributes datasets to allow to add a new column
          X_copy = self.X.copy()

          # define the name of the new column, and then add it to the X_copy
          discrete_class = self.split_attr + '-group'
          X_copy.loc[:,discrete_class] = utils.get_discrete_variables(X_copy,self.split_attr,self.split_point)
          
          # create a child node by the categorical values, such as "petal length" < 2.45 or "petal length" > 2.45
          for value in X_copy[discrete_class].unique():
            index_boolean = X_copy[discrete_class] == value
            self.children[value] = TreeNode(
                              self.X[index_boolean],
                              self.y[index_boolean],
                              node_name = value,
                              default_decision=self.default_decision,)
            # recursively generate the node until reaching the leaf node
            self.children[value].make()
        
        else:
          '''
          Two issuese may happend if split_point is None
          case1: 
          Only one mode means that when attribute values are the same, the frequency of the target value is different.
          For example, if the first target value appears once and the second target value appears three times, 
          then the target value that appears three times wins and can be selected as the decision

          case2:
          if the mode is more than one, it means when the attribute values are the same, the frequency of the target value is identical.
          For example, if the first target value appears once and the second target value appears once
          then it is hard to make a decision.
          This case issue will happen in the tree construction of the first evaluation dataset later.
          '''
          # assign the mode of target values to the y_mode variable
          y_mode = self.y.mode()


          if len(y_mode) == 1:
            better_decision = y_mode[0]
            self.decision = better_decision   
            # print("Issue happens!")
            # print("The attribute values are the same, and the frequency of the target values is different.")
            # print("Input values: ")
            # print(self.X)
            # print("Target values: ")
            # print(self.y)
            # print("Using the majority voting method to make a better decision -> ", self.decision)
            # print("------------------------------------------------------------------------------------------------------------")         

          else:
            decision = y_mode[0]
            self.decision = decision
            # print("Issue happens!")
            # print("The attribute values are the same, but the frequency of the target values is identical")
            # print("Input values: ")
            # print(self.X)
            # print("Target values: ")
            # print(self.y)
            # print("Just make a decision -> ", self.decision)
            # print("------------------------------------------------------------------------------------------------------------")
          return

  def print_node_details(self, vis):
    '''
    To print the details of each node, the output for each node will be:
      1. (First view of the node / Back to the node) : node name
      2. number of samples in this node
      3. The counts of y values
      4. The information entropy in this node
      5. every attributes with its information gain ratio
      6. The details of the splitted point
      7. The node name of the child node.
    
    if the node is the leaf node, the output will be:
      1. The node name
      2. The information entropy
      3. The determined decision 
    '''
    # intialize the time
    time = 1
    if self.split_attr is not None:
      for k,v in self.children.items():
        if len(self.split_point_info) != 0:
          # this if-else clause is to output it is the first time or the second time to view this node
          # if it is the second time to view the node, output "Back to the node"
          if time != 1:
            print("Back to the node: ",self.name)
          else:
            print("First view of the node: ",self.name)

          print("The number of the samples in this node is: ", len(self.X))
          print(self.y.value_counts())
          print("The Entropy is: {}\n".format(self.entropy))
          for i,j in self.attrs_info_gain_ratio.items():          
            print("{}'s information gain ratio is: {}".format(i,j))

          print("\nSplited by {}, its information gain ratio is {}, its splited points list is:\n {}".format(self.split_point_info[1][1],self.split_point_info[1][2],self.split_point_info[0]))
          print("Among these splited points, the best splited point is {}".format(self.split_point_info[1][0]))
          print("The child node is: {}".format(k))
          vis[k] = {
            "root": self.name
          }
          print('--------------------------------------------------------------------')
          time += 1
          v.print_node_details(vis)
          
             
    else:
      time += 1
      print("This node is: {}, it is a leaf node".format(self.name))
      print(self.y.value_counts())
      print("The Entropy is: {}\n".format(self.entropy))
      print("The number of samples is: ", len(self.X))
      print("Determining the decision -> ",self.decision)
      print('--------------------------------------------------------------------')
      vis[self.name]["value"] = int(self.decision)
  
  def print_tree(self, vis):
    root_name= "Root node" 
    print (json.dumps(vis, sort_keys=True, indent=4))