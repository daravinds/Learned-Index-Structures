#!/bin/bash

declare -a datasetSize=(1000000 500000 100000 50000 10000 5000 1000 500 100 50)
errorRate=($(seq 0.1 0.1 0.9))

i=0
for i in "${!datasetSize[@]}"
do
for j in "${!errorRate[@]}"
  do 
  	python NewDeepBloom.py ${datasetSize[$i]} ${errorRate[$j]}
  done
done
