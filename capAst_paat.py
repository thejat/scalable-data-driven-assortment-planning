# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 02:40:04 2016

@author: Deeksha
"""

import numpy
import sys
import time
from capAst_oracle import calcRev

#prod is the number of products
#C is the capacity
#p is a numpy array containing prices of products
#v is the customer preference vector with the first entry denoting the parameter for the no-purchase option
#The code below assumes v to be of length (number of products +1) and p to be of length number of products

#Ensure that the entries of v are distinct otherwise there might be issues
def capAst_paat(prod, C, p, v):
    st = time.time()
    n = prod +1
    v =  v/v[0] #normalizing the no-purchase coefficient to 1 
    if len(p)==prod: 
        p =  numpy.insert(p,0,0) #appending 0 at the beginning to denote 0 price for the no-purchase option
    
    ispt = numpy.empty((n, n))*float("nan") #2-D array for storing intersection points

    for j in range (1,n):
        ispt[0,j] = p[j]
        for i in range (1,j):
            ispt[i,j] = ((p[i]*v[i]) - (p[j]*v[j]))/(v[i] - v[j])
                   
    ix = numpy.argsort(ispt.flatten())        #sorted indexing in the flattened array
    ret = (n*n - n)/2 #number of relevant items, others are all inf
    #pos2 = [[i/n, i%n] for i in ix[:ret]] # stores the sorted indices
    #if three or more lines intersect at a point, then I am not sure how the above ordering would work. This needs to be checked 

    pos = numpy.unravel_index(ix[:ret], numpy.shape(ispt))
    pos = numpy.column_stack((pos[0], pos[1]))

    
    numIspt = len(pos) #storing the number of intersection points
    
    sigma = numpy.ones((n-1, numIspt +1) ) * sys.maxint#the number of possible permutations is 1+ the number of intersection points
    sigma[:,0] = 1+numpy.argsort(-1*v[1:]) #we want to sort v into descending order, so we are sorting -v in ascending order and storing the indexes
    
    A = numpy.ones((C, numIspt +1) ) * sys.maxint
    G = numpy.ones((C, numIspt +1) ) * sys.maxint
    B = numpy.ones((n-1, numIspt +1) ) * sys.maxint
    
    Bcount = -1 #number of elements in current B vector
    
    A[:,0] = sigma[0:C,0]
    G[:,0] = sigma[0:C,0]
    
    for l in range(1,numIspt+1): 
        sigma[:,l] = sigma[:,l-1]
        B[:, l] = B[:, l-1]
        
        #this is to ensure that the first coordinate is smaller -not sure if the below line is foolproof   
        if(pos[l-1][0] > pos[l-1][1]):
#            tmpVar = pos[l-1][1]
#            pos[l-1][1] = pos[l-1][0]
#            pos[l-1][0]  =tmpVar
            pos[l-1][0], pos[l-1][1] = pos[l-1][1], pos[l-1][0]
        #print 'position switched'
        
        if pos[l-1][0] != 0:
            idx1 = numpy.where(sigma[:,l-1] == pos[l-1][0])
            idx2 = numpy.where(sigma[:,l-1] == pos[l-1][1])
            #print sigma[idx1, l-1], sigma[idx2, l-1]
            sigma[idx1, l], sigma[idx2, l] =  sigma[idx2, l-1], sigma[idx1, l-1]
            
        else:
            B[Bcount + 1,l] = pos[l-1][1]
            Bcount = Bcount +1
            
        G[:,l] = sigma[0:C, l]
        temp = numpy.setdiff1d(G[:,l], B[:,l])
        A[0:len(temp), l ] = temp
                
    
    maxRev= 0 #denoting the maximum revenue encountered in the sets till now
    maxRevSet = -1 #denoting which set has the maximum revenue till now
    
    for l in range(numIspt+1):
        objs = A[numpy.where(A[:, l]< sys.maxint), l].flatten()           
        #objs = numpy.where(0 < objs)  
        
    
#        num = 0
#        den = 1
#        for s in range(len(objs)):
#            num  = num + p[int(objs[s])]*v[int(objs[s])]
#            den  = den + v[int(objs[s])]
#            
#        rev = num/den
    
        rev = calcRev(objs.astype('int'), p, v, prod)
        if rev > maxRev:
            maxRev = rev
            maxRevSet = l
            
    
    optSet =  A[numpy.where(A[:, maxRevSet]< sys.maxint), maxRevSet].flatten()
    timeTaken = time.time() - st
    #optSet =  numpy.where(0 < optSet)
    print " " 
    print "Results for Paat's algorithm"
    print "Products in the optimal assortment are", optSet 
    print "Optimal revenue is", maxRev
    print "Time taken for running Paat's algorithm is", timeTaken       
    return maxRev, optSet, timeTaken;       
        
        