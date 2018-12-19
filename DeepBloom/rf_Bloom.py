import numpy as np
import pylab as pl
from sklearn import svm, datasets
import sys
import datetime
import os

def linearly_separable_data():
    X = np.r_[ np.random.randn(20,2) - [3,3], np.random.randn(20,2) + [4,4]]
    X[:,1] = 0
    Y = [0]*20 + [1]*20
    return X, Y


def completely_non_linearly_separable_normal_data(a = 10, gap = 5):
    # row wise merging, 20 points with mean at -3, -3; and so on
    X = gap * np.random.randn(int(a),2) - [1,1]
    print(len(X[:,0]), 'vs', a)
    X[:,1] = 0
    one_label_positions = np.random.choice(np.arange(a), int(a/2), replace=False)
    Y = a*[0]
    for pos in one_label_positions:
        Y[pos] = 1

    return X, Y

def completely_non_linearly_separable_data(a = 100, gap = 5):
    # X = np.random.randint(low=-50, high=50, size=(100, 2))
    # choose 100 unique(w/o replacement) points from within 0 to 150
    temp = np.random.choice(np.arange(int(gap*a)), a, replace=False)
    print(gap)
    print(len(set(temp)), 'vs', a)
    X = np.vstack((
        temp, a * [0]
    )).T

    one_label_positions = np.random.choice(np.arange(a), int(a/2), replace=False)
    Y = a*[0]
    for pos in one_label_positions:
        Y[pos] = 1

    return X, Y


import time

print('====================================Present Run:==============================')
# X, Y = completely_non_linearly_separable_normal_data(int(sys.argv[1]), float(sys.argv[2])) #for normal distribution
X, Y = completely_non_linearly_separable_data(int(sys.argv[1]), float(sys.argv[2])) # for uniform distribution


start = datetime.datetime.now() # BUILD TIME ENDS - Please use a different function to take this in micro seconds as you did earlier

print(type(X), type(Y))
print(X.shape, len(Y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
fpr_b = 0.01
h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter

rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)


#Predicting on the train set
print('RBF: ')
y_pred = rbf_svc.predict(X_train)
y_label = [int(i) for i in y_train]
from sklearn.metrics import classification_report, confusion_matrix
conf_matrix = confusion_matrix(y_label, y_pred)
print(conf_matrix)
print(classification_report(y_label, y_pred))

print()

# False Negatives
list_FN_positions = list()
cnt = 0
for label in y_label:
    if label == 1:
        if( y_pred[cnt] == 0):
            list_FN_positions.append(cnt)
    cnt = cnt + 1

bloom = None
from bloom_filter import BloomFilter
if len(list_FN_positions) > 0:
    bloom = BloomFilter(max_elements=len(list_FN_positions), error_rate=fpr_b)

    for idx in list_FN_positions:
        bloom.add(str(X[idx,0]))

    #Memory usage
    print("Number of bits:", bloom.num_bits_m)

end = datetime.datetime.now() # BUILD TIME ENDS - Please use a different function to take this in micro seconds as you did earlier
print('Model + Trad Bloom Build Time: ', str(end - start))

# serialize model
from joblib import dump, load
filename = 'model_rbf' + str(np.random.randint(1000))
print(filename)
dump(rbf_svc, filename + '.compressed', compress=True)
fName = filename + '.compressed'
size = os.stat(fName).st_size

print ' '
print 'fileSize'

if(bloom is not None):
    size = size +  (bloom.num_bits_m)/8
print size

print ' '
print ' '

# Total Memory Usage would be size of file + no of bits


#Predict the False Negatives: false negative rate = (fn/number of positives) = fn/fn+tp
num_pos = sum(y_label)
num_neg = len(y_label) - num_pos
pred_pos = sum(y_pred)
pred_neg = len(y_pred) - pred_pos
#print(num_pos)
print('False Negatives: ', conf_matrix[1, 0])
print('False Negative Rate: ',(float)(conf_matrix[1, 0]/num_pos))
#After putting into bloom
# 0

#False Positive Rate: FP/Number of negatives(fp + tn)
#print(num_neg)
print('Still predicting on train set: ')
print('False Positives: ', conf_matrix[0, 1])
orig_fp_rate = (float) (conf_matrix[0, 1]/num_neg)
print('False Positive Rate: ', orig_fp_rate)
print('Overall False Positive Rate: ', orig_fp_rate + ( 1 - orig_fp_rate)*fpr_b)

# False Positive Rate again:
print('Predict on test set')
#need to do this on a test set : fp in test set and num_neg in test set
print('RBF: ')
y_pred = rbf_svc.predict(X_test)
y_label = [int(i) for i in y_test]
num_pos = sum(y_label)
num_neg = len(y_label) - num_pos
print(num_neg)
from sklearn.metrics import classification_report, confusion_matrix
conf_matrix = confusion_matrix(y_label, y_pred)
print(conf_matrix)
print(classification_report(y_label, y_pred))


print('False Positives: ', conf_matrix[0, 1])
orig_fp_rate = (float) (conf_matrix[0, 1]/num_neg)
FP =  orig_fp_rate
print('False Positive Rate: ', orig_fp_rate)
print('Overall False Positive Rate: ', orig_fp_rate + ( 1 - orig_fp_rate)*fpr_b)
#Query time for negative will hop to model prediction first and then, to BF. 1 + 1


############# Trad Bloom #####################

print ' '
print ' '
print '############# Trad Bloom #####################'


newX = X_train[:,0]
newY = y_train
newTestY = y_test
newTestX = X_test[:,0]

size = len(newTestY)
if(FP >= 0.0 or FP >= 0.9):
    err_rate = 0.1
else:
    err_rate = FP


before = datetime.datetime.now()

bloom = BloomFilter(max_elements=size, error_rate=err_rate)

after = datetime.datetime.now()

print('Trad Bloom Build Time: ', str(after - before))

for i in range(len(newX)):
    bloom.add(newX[i])

falsePositive = 0;

for i in range(len(newTestY)):
    if(newTestY[i] == 1 and newTestX[i] in bloom):
        falsePositive = falsePositive + 1


n = float(falsePositive) / float(size)
filterSize = float(bloom.num_bits_m)/float(8)

print("Configured False Positive : ", err_rate)
print("Traditional False Positive : ", n)
print("File Size : ", filterSize)


# newX = X[:,0]
# newY = Y
#
# postiveIndexes = []
# negativeIndexes = []
#
# for i in newY:
#     if(i == 1):
#         postiveIndexes.append(i)
#     else:
#         negativeIndexes.append(i)
#
# size = len(postiveIndexes)
# err_rate = 0.1
#
# bloom = BloomFilter(max_elements=size, error_rate=err_rate)
#
# for i in postiveIndexes:
#     bloom.add(newX[i])
#
# falsePositive = 0;
#
# for i in negativeIndexes:
#     if(newX[i] in bloom):
#         falsePositive = falsePositive + 1
#
#
# n = float(falsePositive) / float(size)
#
# print("Traditional False Positive : ", n)
# print("File Size : ", bloom.num_bits_m)
#
#





#
# poly_svc = svm.SVC(kernel='poly', degree=5, C=C).fit(X, Y)
# print('Poly: ')
# y_pred_poly = poly_svc.predict(X)
# print(confusion_matrix(y_label, y_pred_poly))
# print(classification_report(y_label, y_pred_poly))
