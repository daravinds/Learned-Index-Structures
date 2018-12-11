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

max_elem = int(sys.argv[1])
err_rate = float(sys.argv[2])
#
#max_elem = 1000000
#err_rate = 0.2

bloom = BloomFilter(max_elements=max_elem, error_rate=err_rate)


# Test whether the bloom-filter has seen a key:
#assert "test-key" in bloom is False

# Mark the key as seen
#bloom.add("test-key")

# Now check again

with open("dataset.json", "r") as read_file:
    data = json.load(read_file)

before = datetime.datetime.now()

i = 0
for x in data["positives"]:
    if i == max_elem:
        break;
    i = i+1
    bloom.add(str(x))


after = datetime.datetime.now()
delta = after - before

#print 'difference'
#print delta


print("Size:", max_elem)
print("Build time:", delta)

pLen = len(data["positives"])
nLen = len(data["negatives"])

#print pLen
#print nLen

#positiveFP = 0;
#
#for x in range(100):
#    index = random.randint(0, pLen)
#    #print data["positives"][index]
#    if(str(data["positives"][index]) in bloom):
#        positiveFP = positiveFP +1

negativeFP = 0

#for x in range(100):
#    index = random.randint(0, nLen)
#    if(str(data["negatives"][index]) in bloom):
#        negativeFP = negativeFP +1

for x in data["negatives"]:
    if(str(x) in bloom):
        negativeFP = negativeFP +1

#p = float(positiveFP) / float(100)
#print 'positiveFP'
#print "%.2f" % (p)

n = float(negativeFP) / float(nLen)


print 'Average error rate :'
print "%.2f" % (n)
print("Average error rate : %.2f", n)


print 'Projected error rate :'
print("Projected error rate :", err_rate)

print("Number of bits:", bloom.num_bits_m)

print ' '
print ' '

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
