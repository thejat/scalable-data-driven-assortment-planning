# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import random 
import numpy
from capAst_paat import capAst
from random import randint

#import sys

## required inputs
price_range = 10
prod = 5 #number of products
C = 3 #capacity
#n = prod +1#number of options after including the no-purchase option
v = numpy.random.rand(prod+1) # create an array of n uniform random numbers between 0 and 1     
v = numpy.around(v, decimals =2) #round the random numbers to two decimal places
p = numpy.random.uniform(0,price_range, prod) # create an array of n+1 uniform random numbers between 0 and price_range  
p = numpy.around(p, decimals =2)

capAst(prod, C, p, v)
capAst_oracle(prod, C, p, v)


prod = 4
C =2
p = numpy.asarray([9.5, 9, 7, 4.5])
v = numpy.asarray([1, 0.2, 0.6, 0.3, 5.2])
capAst_paat(4,2 , numpy.asarray(p), numpy.asarray(v))

capAst_paat(4,2, p, v)



p= numpy.asarray([0.98,0.88, 0.82, 0.77, 0.71, 0.60, 0.57, 0.16, 0.04, 0.02])
v = numpy.asarray([1, 0.36, 0.84, 0.62, 0.64, 0.8, 0.31, 0.84, 0.78, 0.38, 0.34])
 
capAst_paat(10, 4, p, v) 



for i in range(50):
    price_range = 10
    #prod = randint(1,10) #number of products
    prod = 10
    C = randint(1,prod) #capacity
    #n = prod +1#number of options after including the no-purchase option
    v = numpy.random.rand(prod+1) # create an array of n uniform random numbers between 0 and 1     
    v = numpy.around(v, decimals =2) #round the random numbers to two decimal places
    p = numpy.random.uniform(0,price_range, prod) # create an array of n+1 uniform random numbers between 0 and price_range  
    p = numpy.around(p, decimals =2)
    
    capAst_paat(prod, C, p, v)
    capAst_oracle(prod, C, p, v)
    

p=numpy.asarray([ 9.79,  4.29,  1.57,  5.5 ,  3.68,  9.13,  3.34,  0.87,  2.32])
v = numpy.asarray([ 0.91,  0.06,  0.22,  0.18,  0.83,  0.22,  0.81,  0.8 ,  0.28,  0.58])
#Products in the optimal assortment are [ 1.  2.  3.  4.]
#Optimal revenue is 3.2485
#Products in the optimal assortment are (1, 4, 6)
#Optimal revenue is 4.80754789272


p= numpy.asarray([ 0.74,  6.86,  7.55,  0.  ,  6.68])
v = numpy.asarray([ 0.91,  0.63,  0.23,  0.43,  0.14,  0.71])



prod = 10
C = 5
v = numpy.random.rand(prod+1) # create an array of n uniform random numbers between 0 and 1     
v = numpy.around(v, decimals =2) #round the random numbers to two decimal places
p = numpy.random.uniform(0,price_range, prod) # create an array of n+1 uniform random numbers between 0 and price_range  
p = numpy.around(p, decimals =2)
ep = 0.2
   
capAst_LSH(prod, C, p, v, ep, max(p))
capAst_oracle(prod, C, p, v)