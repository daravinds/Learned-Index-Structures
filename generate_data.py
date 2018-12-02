# Inspired from https://github.com/yangjufo/Learned-Indexes/blob/master/data/create_data.py

import numpy as np
import csv
import random
from enum import Enum
import os

DATA_SIZE = 100000
BLOCK_SIZE = 10

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

def get_data(distribution, size):
	data = []
	if distribution == Distribution.RANDOM:
		data = random.sample(range(size * 2), size)
	elif distribution == Distribution.BINOMIAL:
		data = np.random.binomial(100, 0.5, size)
	elif distribution == Distribution.POISSON:
		data = np.random.poisson(6, size)
	elif distribution == Distribution.EXPONENTIAL:
		data = np.random.exponential(10, size)
	elif distribution == Distribution.LOGNORMAL:
		data = np.random.lognormal(0, 2, size)
	else:
		data = np.random.normal(1000, 100, size)
	return data

def get_sorted_data(distribution, size):
	data = get_data(distribution, size)
	data.sort()
	return data

def generate_data(distribution, data_size = DATA_SIZE):
	data = get_sorted_data(distribution, data_size)

	multiplicant = 1
	if distribution == Distribution.EXPONENTIAL:
		multiplicant = 10000000
	elif distribution == Distribution.LOGNORMAL:
		multiplicant = 10000

	data_path = os.path.join(os.getcwd(), "data", Distribution.to_string(distribution) + ".csv")
	with open(data_path, 'wb') as csv_file:
		csv_writer = csv.writer(csv_file)
		for index, number in enumerate(data):
			csv_writer.writerow([int(number * multiplicant), index / BLOCK_SIZE])

if __name__ == "__main__":
	generate_data(Distribution.BINOMIAL, 100)
	generate_data(Distribution.EXPONENTIAL, 100)
	generate_data(Distribution.LOGNORMAL, 100)
	generate_data(Distribution.NORMAL, 100)
	generate_data(Distribution.POISSON, 100)
	generate_data(Distribution.RANDOM, 100)
