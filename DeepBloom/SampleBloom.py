from bloom_filter import BloomFilter
import json
import datetime
import sys
import random

# instantiate BloomFilter with custom settings,
# max_elements is how many elements you expect the filter to hold.
# error_rate defines accuracy; You can use defaults with
# `BloomFilter()` without any arguments. Following example
# is same as defaults:
bloom = BloomFilter(max_elements=10000, error_rate=0.1)


# Test whether the bloom-filter has seen a key:
#assert "test-key" in bloom is False

# Mark the key as seen
#bloom.add("test-key")

# Now check again

with open("dataset.json", "r") as read_file:
    data = json.load(read_file)

before = datetime.datetime.now()

int i = 0;
for x in data["positives"]:
    #print x
    if i == 100
        break
	bloom.add(str(x))


after = datetime.datetime.now()
delta = after - before

print 'difference'
print delta

pLen = len(data["positives"])
nLen = len(data["negatives"])


positiveFP = 0;

for x in range(100):
	index = random.randint(0, pLen)
	print data["positives"][index]
	if(str(data["positives"][index]) not in bloom):
		positiveFP = positiveFP +1

negativeFP = 0

for x in range(100):
	index = random.randint(0, pLen)
	if(str(data["negatives"][index]) in bloom):
		negativeFP = negativeFP +1




# for x in data["positives"]:
# 	if(str(x) in bloom)
# 		positivesCorrect = positivesCorrect + 1;
# 	else:
# 		positivesNotCorrect = positivesNotCorrect + 1;





#if("test-key1" in bloom):
# 	print 'true'
# else:
	# print 'False'
#assert "test-key" in bloom is True
