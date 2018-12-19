import numpy as np
import pdb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

class LinearModel:
  def __init__(self, model, train_X, train_Y):
    self.model = model
    self.train_X = train_X
    self.train_Y = train_Y
    self.train_X = np.reshape(self.train_X, (len(self.train_Y), -1))
    self.mean_err = 0
    

  def train(self):
    print(str(len(self.train_X)))
    self.model.fit(self.train_X, self.train_Y)

  def predict(self, input_key):
    test_X = [[input_key]]
    return int(round(self.model.predict(test_X)))

  def set_error(self, test_X, test_Y):
    self.mean_err = self.model.score(self.train_X, self.train_Y)