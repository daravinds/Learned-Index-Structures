import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  
from sklearn import metrics 
import matplotlib.pyplot as plt  
# coding=utf-8
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

path = "lognormal_10000.csv"
data = pd.read_csv(path)

X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

# regressor = LinearRegression()  
# regressor = LogisticRegression()
regressor = LogisticRegression(random_state=0, solver='lbfgs')
regressor.fit(X_train, y_train)  

y_pred = regressor.predict(X_test)  


# to display graph
plt.scatter(X, y)
plt.plot(X_test, y_pred, color='blue', linewidth=3)  
plt.title('val vs index')  
plt.xlabel('val')  
plt.ylabel('index')  
plt.show() 

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
print df 

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 

# print (X_test)

# xprd = [[1043], [1044]]


# yprd = regressor.predict(xprd)

# print ('Predicted', yprd)
# print('Actual 9999')