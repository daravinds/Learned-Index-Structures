# Learned-Index-Structures

A comparison of traditional index structures and machine-learning based variants.


### Prerequisite packages:

Requires Python 2.7.x and the following packages

```
pip install tensorflow
pip install pandas
pip install sklearn
```

### Download instructions:

On your terminal execute,

git clone https://github.com/daravinds/Learned-Index-Structures.git

cd Learned-Index-Structures/
git checkout Learned
git branch
// The output should show current branch as Learned

### Instructions to run:

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
