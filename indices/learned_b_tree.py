from __future__ import print_function
import pandas as pd
from trained_nn import TrainedNN, AbstractNN, ParameterPool, set_data_type
from b_tree import BTree
from linear_model import LinearModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import time, gc, json, datetime
import os, sys, getopt, pdb
from enum import Enum

class Distribution(Enum):
	BINOMIAL = 0
	EXPONENTIAL = 1
	LOGNORMAL = 2
	NORMAL = 3
	POISSON = 4
	RANDOM = 5

	@classmethod
	def to_string(cls, val):
		for k, v in vars(cls).iteritems():
			if v == val:
				return k.lower()

# Setting 
BLOCK_SIZE = 500
TOTAL_NUMBER = 10000

# data files
filePath = {
  Distribution.RANDOM: "data/random.csv",
  Distribution.BINOMIAL: "data/binomial.csv",
  Distribution.POISSON: "data/poisson.csv",
  Distribution.EXPONENTIAL: "data/exponential.csv",
  Distribution.NORMAL: "data/normal.csv",
  Distribution.LOGNORMAL: "data/lognormal.csv"
}

# result record path
pathString = {
  Distribution.RANDOM: "Random",
  Distribution.BINOMIAL: "Binomial",
  Distribution.POISSON: "Poisson",
  Distribution.EXPONENTIAL: "Exponential",
  Distribution.NORMAL: "Normal",
  Distribution.LOGNORMAL: "Lognormal"
}

# threshold for train (judge whether stop train and replace with BTree)
thresholdPool = {
  Distribution.RANDOM: [1, 4],
  Distribution.EXPONENTIAL: [55, 10000],
  Distribution.LOGNORMAL: [1, 4]
}   

# whether use threshold to stop train for models in stages
useThresholdPool = {
  Distribution.RANDOM: [True, False],    
  Distribution.EXPONENTIAL: [True, False],
  Distribution.LOGNORMAL: [True, False]
}

# hybrid training structure, 2 stages
def hybrid_training(threshold, use_threshold, stage_nums, core_nums, train_step_nums, batch_size_nums, learning_rate_nums,
                    keep_ratio_nums, train_data_x, train_data_y, test_data_x, test_data_y):
    stage_length = len(stage_nums)
    col_num = stage_nums[1]

    tmp_inputs = [[[] for i in range(col_num)] for i in range(stage_length)]
    tmp_labels = [[[] for i in range(col_num)] for i in range(stage_length)]
    index = [[None for i in range(col_num)] for i in range(stage_length)]
    tmp_inputs[0][0] = train_data_x
    tmp_labels[0][0] = train_data_y
    test_inputs = test_data_x
    for i in range(0, stage_length):
        for j in range(0, stage_nums[i]):
            if len(tmp_labels[i][j]) == 0:
                continue
            inputs = tmp_inputs[i][j]
            labels = []
            test_labels = []
            if i == 0:
                divisor = stage_nums[i + 1] * 1.0 / (TOTAL_NUMBER / BLOCK_SIZE)
                for k in tmp_labels[i][j]:
                    labels.append(int(k * divisor))
                for k in test_data_y:
                    test_labels.append(int(k * divisor))
            else:
                labels = tmp_labels[i][j]
                test_labels = test_data_y              
            tmp_index = TrainedNN(threshold[i], use_threshold[i], core_nums[i], train_step_nums[i], batch_size_nums[i],
                                    learning_rate_nums[i], keep_ratio_nums[i], inputs, labels, test_inputs, test_labels)            
            tmp_index.train()      
            index[i][j] = AbstractNN(tmp_index.get_weights(), tmp_index.get_bias(), core_nums[i], tmp_index.cal_err())
            del tmp_index
            gc.collect()
            if i < stage_length - 1:
                for ind in range(len(tmp_inputs[i][j])):
                    p = index[i][j].predict(tmp_inputs[i][j][ind])                    

                    if p > stage_nums[i + 1] - 1:
                        p = stage_nums[i + 1] - 1
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    for i in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][i] is None:
            continue
        mean_abs_err = index[stage_length - 1][i].mean_err
        print("mean abs err:", mean_abs_err)
        if mean_abs_err > threshold[stage_length - 1]:
            print("Using BTree")
            index[stage_length - 1][i] = BTree(32)
            index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])
    return index

# hybrid linear training structure, 2 stages
def hybrid_linear_training(threshold, stage_nums, train_data_x, train_data_y, test_data_x, test_data_y, model):
    stage_length = len(stage_nums)
    col_num = stage_nums[1]

    tmp_inputs = [[[] for i in range(col_num)] for i in range(stage_length)]
    tmp_labels = [[[] for i in range(col_num)] for i in range(stage_length)]
    index = [[None for i in range(col_num)] for i in range(stage_length)]
    tmp_inputs[0][0] = train_data_x
    tmp_labels[0][0] = train_data_y
    test_inputs = test_data_x
    for i in range(0, stage_length):
        for j in range(0, stage_nums[i]):
            if len(tmp_labels[i][j]) == 0:
                continue
            inputs = tmp_inputs[i][j]
            labels = []
            test_labels = []
            if i == 0:
                # first stage, calculate how many models in next stage
                divisor = stage_nums[i + 1] * 1.0 / (TOTAL_NUMBER / BLOCK_SIZE)
                for k in tmp_labels[i][j]:
                    labels.append(int(k * divisor))
                for k in test_data_y:
                    test_labels.append(int(k * divisor))
            else:
                labels = tmp_labels[i][j]
                test_labels = test_data_y    
            # train model   
            if model == "linear":
                tmp_index = LinearModel(LinearRegression(), inputs, labels)
            elif model == "logistic":
                tmp_index = LinearModel(LogisticRegression(), inputs, labels)

            tmp_index.train()
            tmp_index.set_error(inputs, labels)
            # get parameters in model (weight matrix and bias matrix)      
            index[i][j] = tmp_index
            del tmp_index
            gc.collect()
            if i < stage_length - 1:
                # allocate data into training set for models in next stage
                for ind in range(len(tmp_inputs[i][j])):
                    # pick model in next stage with output of this model
                    p = index[i][j].predict(tmp_inputs[i][j][ind])
                    if p > stage_nums[i + 1] - 1:
                        p = stage_nums[i + 1] - 1

                    # print(p)
                    tmp_inputs[i + 1][p].append(tmp_inputs[i][j][ind])
                    tmp_labels[i + 1][p].append(tmp_labels[i][j][ind])

    for i in range(stage_nums[stage_length - 1]):
        if index[stage_length - 1][i] is None:
            continue
        mean_abs_err = index[stage_length - 1][i].mean_err
        if mean_abs_err > threshold[stage_length - 1]:
            # replace model with BTree if mean error > threshold
            print("Using BTree")
            index[stage_length - 1][i] = BTree(32)
            index[stage_length - 1][i].build(tmp_inputs[stage_length - 1][i], tmp_labels[stage_length - 1][i])
    return index

# main function for training idnex
def train_index(threshold, use_threshold, distribution, path, model):
    data = pd.read_csv(path, header=None)
    train_set_x = []
    train_set_y = []
    test_set_x = []
    test_set_y = []

    set_data_type(distribution)

    if distribution == Distribution.RANDOM:
        parameter = ParameterPool.RANDOM.value
    elif distribution == Distribution.LOGNORMAL:
        parameter = ParameterPool.LOGNORMAL.value
    elif distribution == Distribution.EXPONENTIAL:
        parameter = ParameterPool.EXPONENTIAL.value
    elif distribution == Distribution.NORMAL:
        parameter = ParameterPool.NORMAL.value
    else:
        return
    stage_set = parameter.stage_set
    stage_set[1] = 4
    core_set = parameter.core_set
    train_step_set = parameter.train_step_set
    batch_size_set = parameter.batch_size_set
    learning_rate_set = parameter.learning_rate_set
    keep_ratio_set = parameter.keep_ratio_set

    global TOTAL_NUMBER
    TOTAL_NUMBER = data.shape[0]

    X = data.iloc[:, :-1].values  
    Y = data.iloc[:, 1].values  
    trai_set_x, tes_set_x, train_set_y, test_set_y = train_test_split(X, Y, test_size=0.1, random_state=0)  
    

    for i in trai_set_x:
      train_set_x.append(i[0])

    for i in tes_set_x:
      test_set_x.append(i[0])

    print("*************start Learned NN************")
    print("Start Train")
    start_time = datetime.datetime.now()

    if model == "neural":
        trained_index = hybrid_training(threshold, use_threshold, stage_set, core_set, train_step_set, batch_size_set, learning_rate_set, keep_ratio_set, train_set_x, train_set_y, [], [])
    else:
        trained_index = hybrid_linear_training(threshold, stage_set, train_set_x, train_set_y, [], [], model)
    learn_time = datetime.datetime.now() - start_time

    print("Build Learned NN time ", learn_time)
    err = 0

    start_time = datetime.datetime.now()
    for ind in range(len(test_set_x)):
        pre1 = trained_index[0][0].predict(test_set_x[ind])
        if pre1 > stage_set[1] - 1:
            pre1 = stage_set[1] - 1
        pre2 = trained_index[1][pre1].predict(test_set_x[ind])
        err += abs(pre2 - test_set_y[ind])

    search_time = (datetime.datetime.now() - start_time)
    print("Elements=", str(len(test_set_x)))
    print("Search time ", search_time)
    mean_error = err * 1.0 / len(test_set_x)
    print("mean error = ", mean_error)
    print("*************end Learned NN************\n\n")

    del trained_index
    gc.collect()
    
def show_help_message():
  print('Usage: python learned_b_tree.py -m <Model> -d <Distribution>')
  sys.exit(2)

def main(argv):
  distribution = None
  model = None
  try:
    opts, args = getopt.getopt(argv, "hd:m:")
  except getopt.GetoptError:
    show_help_message()
  for opt, arg in opts:
    arg = str(arg).lower()
    if opt == '-h':
      show_help_message()
    elif opt == '-m':
      if arg == "linear":
        model = "linear"
      elif arg == "logistic":
        model = "logistic"
      elif arg == "neural_net":
        model = "neural"
      else:
        show_help_message()  
    elif opt == '-d':
      if arg == "random":
        distribution = Distribution.RANDOM
      elif arg == "exponential":
        distribution = Distribution.EXPONENTIAL
      elif arg == "lognormal":
        distribution = Distribution.LOGNORMAL
      else:
        show_help_message()
    else:
      show_help_message()

  if not distribution:
    show_help_message()

  train_index(thresholdPool[distribution], useThresholdPool[distribution], distribution, filePath[distribution], model)

if __name__ == "__main__":
    main(sys.argv[1:])
