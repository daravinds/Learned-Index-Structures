import bitarray, math, time
import hashlib
from struct import unpack, pack, calcsize

class BloomFilter(object):
    FILE_FMT = b'<dQQQQ'

    def __init__(self, capacity, error_rate=0.001):
        """Implements a space-efficient probabilistic data structure

        capacity
            this BloomFilter must be able to store at least *capacity* elements
            while maintaining no more than *error_rate* chance of false
            positives
        error_rate
            the error_rate of the filter returning false positives. This
            determines the filters capacity. Inserting more than capacity
            elements greatly increases the chance of false positives.

        >>> b = BloomFilter(capacity=100000, error_rate=0.001)
        >>> b.add("test")
        False
        >>> "test" in b
        True

        """
        if not (0 < error_rate < 1):
            raise ValueError("Error_Rate must be between 0 and 1.")
        if not capacity > 0:
            raise ValueError("Capacity must be > 0")
        num_slices = int(math.ceil(math.log(1.0 / error_rate, 2)))
        print error_rate
        print 1.0 / error_rate
        print 'num_slices' + str(num_slices)

        bits_per_slice = int(math.ceil(
            (capacity * abs(math.log(error_rate))) /
            (num_slices * (math.log(2) ** 2))))

        print 'numerator'
        print capacity * abs(math.log(error_rate))
        print math.log(2) ** 2
        print num_slices * (math.log(2) ** 2)


        self._setup(error_rate, num_slices, bits_per_slice, capacity, 0)
        self.bitarray = bitarray.bitarray(self.num_bits, endian='little')
        self.bitarray.setall(False)

    def make_hashfuncs(self, num_slices, num_bits):
        print 'here'
        print (1 << 31)

        print 'here'
        print (1 << 15)

        if num_bits >= (1 << 31):
            fmt_code, chunk_size = 'Q', 8
        elif num_bits >= (1 << 15):
            fmt_code, chunk_size = 'I', 4
        else:
            fmt_code, chunk_size = 'H', 2

        print 'fmt_code'
        print fmt_code

        total_hash_bits = 8 * num_slices * chunk_size
        if total_hash_bits > 384:
            hashfn = hashlib.sha512
        elif total_hash_bits > 256:
            hashfn = hashlib.sha384
        elif total_hash_bits > 160:
            hashfn = hashlib.sha256
        elif total_hash_bits > 128:
            hashfn = hashlib.sha1
        else:
            hashfn = hashlib.md5
        fmt = fmt_code * (hashfn().digest_size // chunk_size)
        num_salts, extra = divmod(num_slices, len(fmt))
        if extra:
            num_salts += 1
        salts = tuple(hashfn(hashfn(pack('I', i)).digest()) for i in range(num_salts))
        def _make_hashfuncs(key):
            if running_python_3:
                if isinstance(key, str):
                    key = key.encode('utf-8')
                else:
                    key = str(key).encode('utf-8')
            else:
                if isinstance(key, unicode):
                    key = key.encode('utf-8')
                else:
                    key = str(key)
            i = 0
            for salt in salts:
                h = salt.copy()
                h.update(key)
                for uint in unpack(fmt, h.digest()):
                    yield uint % num_bits
                    i += 1
                    if i >= num_slices:
                        return


        return _make_hashfuncs


    def _setup(self, error_rate, num_slices, bits_per_slice, capacity, count):
            self.error_rate = error_rate
            self.num_slices = num_slices
            self.bits_per_slice = bits_per_slice
            self.capacity = capacity
            self.num_bits = num_slices * bits_per_slice
            self.count = count
            self.make_hashes = self.make_hashfuncs(self.num_slices, self.bits_per_slice)

    

    def add(self, key, skip_check=False):
        """ Adds a key to this bloom filter. If the key already exists in this
        filter it will return True. Otherwise False.

        >>> b = BloomFilter(capacity=100)
        >>> b.add("hello")
        False
        >>> b.add("hello")
        True
        >>> b.count
        1

        """
        bitarray = self.bitarray
        bits_per_slice = self.bits_per_slice
        hashes = self.make_hashes(key)
        hashes = []
        found_all_bits = True
        if self.count > self.capacity:
            raise IndexError("BloomFilter is at capacity")
        offset = 0
        for k in hashes:
            if not skip_check and found_all_bits and not bitarray[offset + k]:
                found_all_bits = False
            self.bitarray[offset + k] = True
            offset += bits_per_slice

        if skip_check:
            self.count += 1
            return False
        elif not found_all_bits:
            self.count += 1
            return False
        else:
            return True

def main(capacity=100000, request_error_rate=0.1):
    print "inside"
    f = BloomFilter(capacity=capacity, error_rate=request_error_rate)
    # assert (capacity == f.capacity)
    start = time.time()
    for i in range(f.capacity):
        #print i
        f.add(i, skip_check=True)
    end = time.time()
    i = 1
    print(i in f)
    print("{:5.3f} seconds to add to capacity, {:10.2f} entries/second".format(
            end - start, f.capacity / (end - start)))
    oneBits = f.bitarray.count(True)
    zeroBits = f.bitarray.count(False)
    #print "Number of 1 bits:", oneBits
    #print "Number of 0 bits:", zeroBits
    print("Number of Filter Bits:", f.num_bits)
    print("Number of slices:", f.num_slices)
    print("Bits per slice:", f.bits_per_slice)
    print("------")
    print("Fraction of 1 bits at capacity: {:5.3f}".format(
            oneBits / float(f.num_bits)))
    # Look for false positives and measure the actual fp rate
    trials = f.capacity
    fp = 0
    start = time.time()
    for i in range(f.capacity, f.capacity + trials + 1):
        # if i in f:
        fp += 1
    end = time.time()
    print(("{:5.3f} seconds to check false positives, "
           "{:10.2f} checks/second".format(end - start, trials / (end - start))))
    print("Requested FP rate: {:2.4f}".format(request_error_rate))
    print("Experimental false positive rate: {:2.4f}".format(fp / float(trials)))
    # Compute theoretical fp max (Goel/Gupta)
    k = f.num_slices
    m = f.num_bits
    n = f.capacity
    fp_theory = math.pow((1 - math.exp(-k * (n + 0.5) / (m - 1))), k)
    print("Projected FP rate (Goel/Gupta): {:2.6f}".format(fp_theory))

if __name__ == '__main__' :
    status = main()
    # sys.exit(status)
