import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
# %matplotlib inline 


def gen_build_time():
    n_groups = 3

    bt = (0.61562, 0.6355, 0.6414)
    linreg = (0.001576, 0.002507, 0.00092)
    logreg = (1.4632, 0.5389, 1.1694)
    hyblin = (0.35035, 0.34325, 0.46344)
    hyblog = (2.0714, 0.87805, 1.7252)
    hybnn = (0.3865, 0.3881, 0.4462)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.3

    opacity = 1
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index - bar_width, bt, bar_width / 2,
                  alpha=opacity, color='blue', error_kw=error_config,
                  label='B-Tree')

    rects2 = ax.bar(index - bar_width / 2, linreg, bar_width / 2,
                  alpha=opacity, color='red', error_kw=error_config,
                  label='Linear Regression')

    rects3 = ax.bar(index, logreg, bar_width / 2,
                  alpha=opacity, color='green', error_kw=error_config,
                  label='Logistic Regression')

    rects4 = ax.bar(index + bar_width / 2, hyblin, bar_width / 2,
                  alpha=opacity, color='yellow', error_kw=error_config,
                  label='Hybrid - linear regression')

    rects5 = ax.bar(index + bar_width, hyblog, bar_width / 2,
                  alpha=opacity, color='orange', error_kw=error_config,
                  label='Hybrid - logistic regression')

    rects6 = ax.bar(index + bar_width * 3 / 2, hybnn, bar_width / 2,
                  alpha=opacity, color='black', error_kw=error_config,
                  label='Hybrid - Neural Network')

    ax.set_xlabel('Distribution')
    ax.set_ylabel('Build time (in seconds)')
    ax.set_yscale("log", nonposy='clip')
    ax.set_title('Build time for each index (distribution-wise)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Log-normal', 'Exponential', 'Random'))
    ax.legend(loc='upper right', prop={'size': 8})

    plt.show()


def gen_search_time():
    n_groups = 3

    bt = (0.0306, 0.03606, 0.03837)
    linreg = (0.000361, 0.000563, 0.000059)
    logreg = (0.001708, 0.0046, 0.00132)
    hyblin = (0.05202, 0.05185, 0.05431)
    hyblog = (0.05556, 0.05642, 0.0302)
    hybnn = (0.9086, 0.9183, 0.9829)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.3

    opacity = 1
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index - bar_width, bt, bar_width / 2,
                  alpha=opacity, color='blue', error_kw=error_config,
                  label='B-Tree')

    rects2 = ax.bar(index - bar_width / 2, linreg, bar_width / 2,
                  alpha=opacity, color='red', error_kw=error_config,
                  label='Linear Regression')

    rects3 = ax.bar(index, logreg, bar_width / 2,
                  alpha=opacity, color='green', error_kw=error_config,
                  label='Logistic Regression')

    rects4 = ax.bar(index + bar_width / 2, hyblin, bar_width / 2,
                  alpha=opacity, color='yellow', error_kw=error_config,
                  label='Hybrid - linear regression')

    rects5 = ax.bar(index + bar_width, hyblog, bar_width / 2,
                  alpha=opacity, color='orange', error_kw=error_config,
                  label='Hybrid - logistic regression')

    rects6 = ax.bar(index + bar_width * 3 / 2, hybnn, bar_width / 2,
                  alpha=opacity, color='black', error_kw=error_config,
                  label='Hybrid - Neural Network')

    ax.set_xlabel('Distribution')
    ax.set_ylabel('Search time (in seconds)')
    ax.set_yscale("log", nonposy='clip')
    ax.set_title('Search time for each index (distribution-wise)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Log-normal', 'Exponential', 'Random'))
#     ax.legend(loc='upper right', prop={'size': 4})

    plt.show()

def gen_error_rate():
    n_groups = 3

    bt = (0, 0, 0)
    linreg = (23.904, 11.371, 0.2571)
    logreg = (20.585, 48.727, 14.867)
    hyblin = (23.829, 11.321, 0.069)
    hyblog = (19.497, 48.754, 14.76)
    hybnn = (230.37, 117.13, 1.68)

    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.3

    opacity = 1
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index - bar_width, bt, bar_width / 2,
                  alpha=opacity, color='blue', error_kw=error_config,
                  label='B-Tree')

    rects2 = ax.bar(index - bar_width / 2, linreg, bar_width / 2,
                  alpha=opacity, color='red', error_kw=error_config,
                  label='Linear Regression')

    rects3 = ax.bar(index, logreg, bar_width / 2,
                  alpha=opacity, color='green', error_kw=error_config,
                  label='Logistic Regression')

    rects4 = ax.bar(index + bar_width / 2, hyblin, bar_width / 2,
                  alpha=opacity, color='yellow', error_kw=error_config,
                  label='Hybrid - linear regression')

    rects5 = ax.bar(index + bar_width, hyblog, bar_width / 2,
                  alpha=opacity, color='orange', error_kw=error_config,
                  label='Hybrid - logistic regression')

    rects6 = ax.bar(index + bar_width * 3 / 2, hybnn, bar_width / 2,
                  alpha=opacity, color='black', error_kw=error_config,
                  label='Hybrid - Neural Network')

    ax.set_xlabel('Distribution')
    ax.set_ylabel('Error rate (Mean absolute error)')
    ax.set_yscale("log", nonposy='clip')
    ax.set_title('Error rate for each index (distribution-wise)')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('Log-normal', 'Exponential', 'Random'))
#     ax.legend(loc='upper right', prop={'size': 4})

    plt.show()

if __name__ == "__main__":
    gen_build_time()
    # gen_search_time()
    # gen_error_rate()