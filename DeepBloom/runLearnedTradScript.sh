#!/bin/bash

declare -a datasetSize=(1000000 500000 100000 50000 10000 5000 1000 500 100 50)

i=0
for i in "${!datasetSize[@]}"
do
	python rf_Bloom.py ${datasetSize[$i]} 5
done
