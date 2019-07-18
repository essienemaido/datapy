# datapy
Algorithms
We have learned a bit about how to program in Python and some ways in which we can make our code more Pythonic. However, programming is not only about making the computer do work for us, its about optimizing the amount of work the computer needs to do. There are multiple types of work we can consider, but here we will consider three major bottlenecks in code:

Computational Complexity - how many instructions are executed?
Memory Needs - how much memory is needed?
I/O - How many reads and writes or network requests do I need to make?
An Algorithm is a procedure for solving a problem. It describes a sequence of operations then when performed will result in a solution to a problem. There are many types of algorithms, some are guaranteed find a solution, some do not. Often we are interested in understanding the performance of of an algorithm in terms of the three bottlenecks listed above (as well as others). In order to analyze these algorithms, we need to develop some tools to understand how algorithms behave as a function of the problem size.

Big O
In order to quantify the complexity of a particular algorithm, we can consider how the algorithm grows with respect to the size of the problem. For the purposes of this notebook we will only consider problems that are one dimensional, so we can quantify the algorithm with respect to a single number, which we will denote as NN. Remember that a problem itself does not have a complexity, rather it is the algorithmic solution which has complexity. For example, lets consider the problem of summing all the numbers between 1 and NN (inclusive). On way to sum this might be to take the of all of these numbers.

def sum_num(N):
    sum_ = 0
    for n in range(N + 1):
        sum_ += n
    return sum_
This algorithm will be O(N)O(N) because we need to perform about NN operations. Note that we only care about the dominant function of NN in the expansion so for our purposes O(N)≈O(N+1)≈O(2N)O(N)≈O(N+1)≈O(2N).

However, if we remember think a bit about how numbers sum, we can invoke a summation rule often attributed to Gauss which says that
∑n=1Nn=N(N+1)2
∑n=1Nn=N(N+1)2
def sum_gauss(N):
    return N*(N+1)//2 # We can use integer division here, why?
This algorithm is O(1)O(1) because it does not depend on how the size of NN!. Lets just check that it gives the same answer.

for N in range(100):
    assert sum_num(N) == sum_gauss(N)
Now lets plot the time it takes to compute these functions as a function of NN. We will use a package called matplotlib to do some plotting, don't worry, we will learn about it later!

We will time how long it takes to perform both of these algorithms. We will take the mean of several runs.

import matplotlib.pyplot as plt
import time
​
def compute(n_avgs, func, N):
    times = []
    for _ in range(n_avgs):
        ts = time.time()
        func(N)
        times.append(time.time() - ts)
    return sum(times)/float(len(times)) * 1000 # milliseconds
​
n_avgs = 100
time_sum = []
time_gauss = []
N_range = range(10,100000, 5000)
for N in N_range:
    time_sum.append(compute(n_avgs, sum_num, N))
    time_gauss.append(compute(n_avgs, sum_gauss, N))
plt.plot(N_range, time_sum, 'o-', label='Sum Numbers')
plt.plot(N_range, time_gauss, 'o-', label='Gauss')
plt.xlabel('N')
plt.ylabel('Average time (ms)')
plt.legend()
Computational Complexity
Lets solve a version of a common problem you might find as a data scientist, how should I store my data? Lets take a very simple case where our data is just a list of numbers and we need to store this in a list? In there any way to optimize the storage?

Lets consider the tradeoffs for various things we might want to do in the list.

Finding an element
If we want to find an element in a list and we know nothing about that list, then we need to check every element in the list to see if that element is there. Lets write a function to do this.

def find_ele(list_, ele):
    for i in list_:
        if i == ele:
            return True
    return False
In order to test these, lets use the random module to generate a list of random numbers between 00 and 10∗N10∗N where NN is the length of the list we want.

import random
def random_list(N, sort=False):
    list_ = [random.randint(0, 10*N) for _ in range(N)]
    return sorted(list_) if sort else list_
random_list(5)
import numpy as np
​
def time_func(func, *args):
    ts = time.time()
    func(*args)
    return time.time() - ts
​
def compute_with_list(n_avgs, N, sort, *funcs):
    ans = []
    for _ in range(n_avgs):
        list_r = random_list(N, sort)
        n_to_find = random.randint(0, 10*N)
        ans.append([time_func(func, list_r, n_to_find)
                for func in funcs])
    # now find avg
    return np.array(ans).mean(axis=0)*1000
    
​
n_avgs = 40
N_range = range(10, 100000, 10000)
time_list = np.array([compute_with_list(n_avgs, N, False, find_ele) for N in N_range])
plt.plot(N_range, time_list, 'o-')
Let us take a slightly different approach where we know that this list sorted. Note that sorting itself is Nlog(N)Nlog⁡(N) complexity, so although we will be able to perform optimized searches on a sorted list, its not in general faster to sort and then find the elements. However, if we know we will be searching often, we can build up the list as a sorted structure and for now we can assume that we have already done so.

The most basic optimization we can perform is to only check until we have seen a number greater than what we are looking for. Since we know the list is sorted, we are guaranteed to not find the number in the rest of the list.

def find_ele_sorted(list_, ele):
    for i in list_:
        if i == ele:
            return True
        if i > ele:
            return False
    return False
​
n_avgs = 40
N_range = range(10, 100000, 10000)
time_list = np.array([compute_with_list(n_avgs, N, True, find_ele, find_ele_sorted) for N in N_range])
plt.plot(N_range, time_list[:,0], 'o-', label='find_ele')
plt.plot(N_range, time_list[:,1], 'o-', label='find_ele_sorted')
plt.legend()
This does better on average, but it still has the same O(N)O(N) runtime. Such optimizations are useful, but we can do better. Lets implement what is sometimes known as binary search. This is a recursive algorithm that allows the list to be divided roughly in half on each recursive step. this will yield logarithmic asymptotic run time. Lets first illustrate the algorithm by walking through an example where l_=[1,2,3,4,5,6,7,8,9,10,11] and we want to check if 2 is contained in the list.

First we check the midpoint of the list, which is 6. We know that 2 does not equal 6, but since the list is sorted, we can immediately rule out the part of the list containing numbers greater than 6. Thus we have already ruled out half the elements of the list.

Now we can ask the question is 2 contained in list [1,2,3,4,5]. First we check the midpoint element of the list, which is 3. We know that 3 is not 2, but again, since 3>23>2, we can eliminate half the list.

Now we can check if 2 is contained in the list [1,2]. We will take midpoint of this list as the first element (since it has index 1=len(list)/21=len(list)/2), and this is equal to 2. Thus 2 is in the original list.

We can see we have performed this search in only three steps and up to an extra step, this did not depend on where 2 was in the list, only that it was sorted. Since we are removing half the list each time, we expect that the number of steps will be roughly log(N)log(N), where the logarithm is understood to be base 2. Lets make a plot of this function compared to NN.

x = np.linspace(10, 2000, 200)
plt.plot(x, np.log(x)/x)
plt.xlabel('N')
plt.ylabel(r'$\log(x)/x$')
Now we can compare this to our other search algorithms.

def find_ele_binary(l_, ele):
    if len(l_) < 1:
        return False
    mid_point = len(l_)//2
    if l_[mid_point] == ele:
        return True
    elif l_[mid_point] > ele:
        return find_ele_binary(l_[:mid_point], ele)
    else:
        return find_ele_binary(l_[mid_point+1:], ele)
n_avgs = 50
N_range = np.arange(1000, 70000, 8000)
time_list = np.array([compute_with_list(n_avgs, N, True, find_ele_sorted, find_ele_binary) for N in N_range])
for i, func in enumerate(['find_ele_sorted', 'find_ele_binary']):
    l, = plt.plot(N_range, 2**time_list[:, i], 'o-', label=func)
    # fit a line to the exponent
    p = np.polyfit(N_range, 2**time_list[:, i], 1)
    plt.plot(N_range, N_range * p[0] + p[1], color=l.get_color())
​
plt.legend()
Of course, if we are only keeping track of what numbers we have seen, we can use something like a set which will be O(1)O(1) access.

Memoization
Often we can get a performance increase just by not recomputing things we have already computed! Let's look again at our recursive Fibonacci sequence defined in a previous notebook.

def fibonacci_recursive(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n-1)  + fibonacci_recursive(n-2)
Lets make a slightly different version which keeps track of how many times we call the function on each element.

from collections import defaultdict
def fibonacci_count(n, d):
    d[n] += 1
    if n == 0:
        return 0, d
    elif n == 1:
        return 1, d
    else:
        n1, _ = fibonacci_count(n-1, d)
        n2, _ = fibonacci_count(n-2, d)
        return n1 + n2, d
Lets see this in action for N=5N=5.

N = 5
ans, d = fibonacci_count(N, defaultdict(int))
for i in range(N):
    print(i, d[i])
          5
      4       3
     3 2     2 1
   2 1 1 0  1 0
  1 0
Now lets look for N=25N=25.

N = 25
ans, d = fibonacci_count(N, defaultdict(int))
print(ans)
for i in range(N):
    print(i, d[i])
Notice that we are calling some of these functions with the same argument thousands of time. If we store the answer to the problem instead of recomputing it, can we do any better?

def fibonacci_mem(n, d):
    if n in d:
        return d[n]
    elif n == 0:
        ans = 0
    elif n == 1:
        ans = 1
    else:
        ans = fibonacci_mem(n-1, d) + fibonacci_mem(n-2, d)
    d[n] = ans
    return ans
%%timeit
fibonacci_mem(33, {0:0,1:1})
%%timeit
fibonacci_recursive(33)
fibonacci_mem(33, {}) == fibonacci_recursive(33)
Our memoized solution does much better, it is several orders of magnitude faster than the bare recursive solution.

However, it does come at a cost, although we save computation, we must use more memory to store the previous result. Often there will be a tradeoff between the two.

Exercise
Write the factorial function f(n)=n!f(n)=n! as a recursive function.
Would memoization make this function faster?
Now what if we needed to calculate the factorial often (perhaps we were computing probabilities of different selections), would memoization be useful in this case?
Memory
As seen before memoization has a tradeoff in terms of memory. Lets try to describe that here for the case of the Fibonacci sequence. We have to keep track of a single element number (the computed solution) for all number less than NN, the number we want to compute. Thus the memory we need grows with problem size as O(N)O(N).

We can analyze our algorithms in terms of memory in a similar way. Again remember, it is the algorithm (and its implementation) which has memory complexity, not the problem itself.

For our first problem, we will again look at summing the numbers between 0 and NN, and we will take two different approaches.

For the first we will build a list of these elements and then sum them.

def sum_list(n):
    numbers = range(n)
    return sum(numbers)
def sum_iter(n):
    number = 0
    sum_ = 0
    while number < n:
        sum_ += number
        number += 1
    return sum_
sum_list(100), sum_iter(100)
Choose a data structure wisely
As we may have noticed in the sorting section, the type of data structure we use is often tied into our choice of algorithm. For example, if we don't already have sorted data, we probably don't want to use binary search because we would need to sort the data first and then would negate any search improvement (sorting is worse than  O(N)O(N) ).

This can be mitigated by choosing our original structure wisely, especially when get to build it from raw data. For example when building a list, inserting elements in a sorted manner can be done in  O(log(N))O(log(N))  time (with almost the same as binary search).

Other data structures lend themselves to other algorithmic purposes.. For example, a heap (implemented in Python with the heapq library) implements a tree like structure which is useful for order statistics, such as keeping track of the largest or smallest  NN  items in a collection. You can read more about it here.

Even as you work through your miniprojects, sometimes choosing a dictionary instead of a list will be the difference between minutes or seconds of computation.

Exercises
Explain why sorting and then using binary search is slower than just searching.
Implement insertion on a list using the same principles as binary search.
