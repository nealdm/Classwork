# profiling.py
"""Profiling.
<Neal Munson>
<1/8/2019>
"""

import numpy as np
import math
from numba import jit
import timeit
from matplotlib import pyplot as plt

def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    '''This will be done by going from the bottom to the top, replacing
    each row of values with itself plus the greater of it's two children.'''
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    nrows = len(data)
    for row in range(nrows - 1)[::-1]: # this selects the rows from bottom to top, starting
        for c in range(len(data[row])):                         # at the second to last row
            data[row][c] = data[row][c] + max(data[row + 1][c], data[row + 1][c+1])  # compute new values for the row
    return(data[0][0])


def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = []
    numprimes = 0  # numprimes is established so the while statement need not
    if N > 0:      # do excessive funcitons, though introducing this counter leads
        primes_list.append(2)            # to needing other simpler lines of code
        numprimes = 1                    # to be added.
    current = 3
    while numprimes < N:
        isprime = True
        sqrt = math.sqrt(current)
        for i in primes_list:     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
                break
            if i > sqrt:
                break
        if isprime:
            primes_list.append(current)
            numprimes += 1
        current += 2
    return primes_list

def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x,
    without using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return(np.argmin(np.linalg.norm( (A-x.reshape((-1,1))) ,axis=0) )) # computes the
                                       # whole thing on a single line.  Note that the
                                       # reshaping of the vector determines which direction
                                       # the vector is subtracted.

def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        # create a dictionary for the alphabet
    dic = {let:i+1 for i,let in enumerate(alphabet)}
                                        # create a dictionary for the names'
                                        # alphabetical value to use to
                                        # calculate 'namescore'
    alphvals={letters:sum(dic[i] for i in letters) for letters in names}
                                        # calculate the sum of all the namescores
                                        # by iterating over each name and looking
                                        # for the appropriate values to use in
                                        # calculation in the dictionary
    return sum([(pos+1)*alphvals[names[pos]] for pos,name in enumerate(names)])


def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    x0 = 0
    x1 = 1
    while True:
        x0,x1 = x1,x0+x1  # although this does not 'index' the numbers
        yield x0          # the same way the book has them, it does return
                          # the right answers (indexing is just shifted by 1)

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for i,num in enumerate(fibonacci()):
        if len(str(num)) == N:
            return i+1  # this returns the index as shifted by one because
                        # the way the lab manual has the fibonacci sequences
                        # indexed is one off of the way I have it indexed.


def prime_sieve(N):
    """Yield all primes that are less than N."""
    ints = np.arange(2,N+1)
    while ints.size != 0:
        first = ints[0]
        ints = ints[ ints % first !=0 ] # removes integers that are divisible by
                                    # the first entry in the list
        yield first


def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit(nopython=True)
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    times =[]
    nums = []
    smallmatrix = np.random.random((2,2))
    matrix_power_numba(smallmatrix,2)
    for m in range(2,8):
        pow = 2**m
        nums.append(pow)
        A = np.random.random([pow,pow])
        start1 = timeit.default_timer()
        matrix_power(A,n)
        t1 = timeit.default_timer() - start1
        start2 = timeit.default_timer()
        matrix_power_numba(A,n)
        t2 = timeit.default_timer() - start2
        start3 = timeit.default_timer()
        np.linalg.matrix_power(A,n)
        t3 = timeit.default_timer() - start3
        times.append([t1,t2,t3])
    times = np.array(times)
    nums = np.array(nums)
    print(nums)
    print(times[:,0])
    print(times[:,1])
    # plt.subplot(2,1,1)
    plt.loglog(nums,times[:,0],label="matrix power")
    plt.loglog(nums,times[:,1],label="matrix power numba")
    # plt.subplot(2,1,2)
    plt.loglog(nums,times[:,2],label="linalg matrix power")
    plt.legend()
    plt.show()
