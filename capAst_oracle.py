# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 05:13:14 2016

@author: Deeksha
"""

from itertools import combinations
import numpy as np
import time

def calcRev(ast, p, v, prod):
#v and p are expected to be n+1 and n length lists respectively 
    if len(p)==prod:    
        p =  np.insert(p,0,0)   #making p a n+1 length list by inserting a 0 in the beginning
    num = 0
    den = v[0]
    for s in range(len(ast)):
        num  = num + p[ast[s]]*v[ast[s]]
        den  = den + v[ast[s]]
    rev = num/den
    return rev

def capAst_oracle(prod, C, p, v,meta=None):
    maxRev= 0 #denoting the maximum revenue encountered in the sets till now
    maxRevSet = -1 #denoting which set has the maximum revenue till now
    astList = list(combinations(range(1,prod+1), 1))
    for i in range(2, C+1):
        astList = astList + list(combinations(range(1,prod+1), i))
    st = time.time()    
    for ast in astList:
        rev = calcRev(ast, p, v, prod)
        if rev > maxRev:
            maxRev = rev
            maxRevSet = ast
    timeTaken = time.time() - st      
    
    # print " " 
    # print "Results for oracle"
    print "Products in the optimal assortment are", maxRevSet 
    # print "Optimal revenue is", maxRev
    # print 'Time taken for running the oracle is', timeTaken
    
    return maxRev, maxRevSet,timeTaken
        

def genAst_oracle(prod,C,p,v,meta=None):

    feasibles = meta['feasibles']

    maxRev= 0 #denoting the maximum revenue encountered in the sets till now
    maxRevSet = -1 #denoting which set has the maximum revenue till now
    st = time.time()    
    for ast0 in feasibles:
        # print 'ast0',ast0
        # print 'prod',prod
        # print 'v',v
        # print 'p',p
        ast = []
        for temp1,temp2 in enumerate(ast0):
            if temp2 ==1:
                ast.append(temp1+1)
        ast = tuple(ast)
        # print 'ast',ast
        rev = calcRev(ast, p, v, prod)
        if rev > maxRev:
            maxRev = rev
            maxRevSet = ast
    timeTaken = time.time() - st      
    
    # print "Products in the optimal assortment are", maxRevSet 
    # print "Optimal revenue is", maxRev
    # print 'Time taken for running the oracle is', timeTaken
    
    return maxRev, set(maxRevSet),timeTaken
        