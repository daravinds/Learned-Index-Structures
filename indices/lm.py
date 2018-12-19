import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import matplotlib.pyplot as plt  
# coding=utf-8
import warnings, sys, datetime
warnings.filterwarnings("ignore", category=FutureWarning)

path = sys.argv[1]
data = pd.read_csv(path)

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)  

start = datetime.datetime.now()

regressor = LogisticRegression()
# regressor = LogisticRegression()
#regressor = LogisticRegression(random_state=0, solver='lbfgs')
regressor.fit(X_train, y_train)  

build = datetime.datetime.now() - start

start = datetime.datetime.now()
y_pred = regressor.predict(X_test)  

lookup = datetime.datetime.now() - start


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print df 

print("Build time: " + str(build))
print("Lookup time: " + str(lookup))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

# print (X_test)

# xprd = [[1043], [1044]]


# yprd = regressor.predict(xprd)

# print ('Predicted', yprd)
# print('Actual 9999')
