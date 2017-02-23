# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:54:24 2016

@author: Deeksha
"""
import numpy
from random import randint

def exactNN(dataset, queryset):
    prod_ = numpy.empty(len(dataset));    
    for i in range(len(dataset)):
        prod_[i] = numpy.inner(dataset[i], queryset)
    ix = numpy.argsort(prod_)
    ix = ix[::-1]
    print ix
    
    
    #usage to check approxNN answers
#    ss = randint(1, 10)
#    ds = numpy.random.rand(20,ss)
#    q = numpy.random.rand(1,ss)
#    test_approxNN("mips", ds, q, 0.8, 2, 0.9)
#    exactNN(ds,q)