# Learned-Index-Structures

A comparison of traditional index structures and machine-learning based variants.


### Prerequisite packages:

_B-Tree Variants_
Requires Python 2.7.x and the following packages

```
pip install tensorflow
pip install pandas
pip install sklearn
```
_Bloom Filter Variants_
Requires Python 3.5.x and the following packages

```
pip install numpy
pip install scikit-learn
pip install bloom-filter
pip install joblib
pip install matplotlib
```

### Download instructions:

On your terminal execute,

```
git clone https://github.com/daravinds/Learned-Index-Structures.git

cd Learned-Index-Structures/
git branch
// The output should show current branch as master
```
### Instructions to run:

_B-Tree Variants_

1. B-Tree index

python indices/b_tree.py <filepath> <pagesize>

*Example*:
	python indices/b_tree.py data/exponential.csv 64	

2. Linear regression index

`python indices/linear_regression.py <filepath>`

*Example*:
	`python indices/linear_regression.py data/random.csv`

3. Logistic regression index

`python indices/logistic_regression.py <filepath>`

*Example*:
	`python indices/logistic_regression.py data/random.csv`

4. Hybrid linear regression index

`python indices/learned_b_tree.py -d <distribution> -m linear`

*Example*:
	`python indices/learned_b_tree.py -d lognormal -m linear`

5. Hybrid logistic regression index

`python indices/learned_b_tree.py -d <distribution> -m logistic`

*Example*:
	`python indices/learned_b_tree.py -d lognormal -m logistic`

6. Hybrid neural network index

`python indices/learned_b_tree.py -d <distribution> -m neural_net`

*Example*:
	`python indices/learned_b_tree.py -d exponential -m neural_net`


_Bloom Filter Variants_


Both, the learned version of Bloom Filter and the traditional Bloom filter will be run on the same dataset using the below script. The dataset is picked out of a uniform distribution of integers. The size and spread of distribution can be controlled from the arguments given to the script.


`python rbf_Model_BF.py <size_of_dataset> <spread> <seed>`

size_of_dataset - is the size of the dataset that you want to run the Bloom filter variations for.
spread -  spread is the range of data that size_of_dataset points will be picked from.
seed - integer for reproducible results

Examples:

```
python rbf_Model_BF.py 100 5 2 
# means choose 100 unique (without replacement) points from points lying in [0, 500). 2 is the seed number for reproducing older results if you need to.

python rbf_Model_BF.py 1000 2 30
# choose 1000 unique (without replacement) points from points lying in [0, 2000) with the seed number of 30
```