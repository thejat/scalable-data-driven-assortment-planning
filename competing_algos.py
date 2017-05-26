# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 05:13:14 2016

@author: Deeksha
"""

from itertools import combinations
import numpy as np
import time, sys
from gurobipy import *

######### ORACLE #################################

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
    
    print "capAst_oracle: Products in the optimal assortment are", maxRevSet 
    
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
    
    print "genAst_oracle: Products in the optimal assortment are", maxRevSet 
    
    return maxRev, set(maxRevSet),timeTaken
        

######### ORACLE #################################



######### STATIC-MNL #################################

#prod is the number of products
#C is the capacity
#p is a np array containing prices of products
#v is the customer preference vector with the first entry denoting the parameter for the no-purchase option
#The code below assumes v to be of length (number of products +1) and p to be of length number of products

#Ensure that the entries of v are distinct otherwise there might be issues
def capAst_paat0(prod, C, p, v):
    st = time.time()
    n = prod +1
    v =  v/v[0] #normalizing the no-purchase coefficient to 1 
    if len(p)==prod: 
        p =  np.insert(p,0,0) #appending 0 at the beginning to denote 0 price for the no-purchase option
    
    ispt = np.empty((n, n))*float("nan") #2-D array for storing intersection points

    for j in range (1,n):
        ispt[0,j] = p[j]
        for i in range (1,j):
            ispt[i,j] = ((p[i]*v[i]) - (p[j]*v[j]))/(v[i] - v[j])
                   
    ix = np.argsort(ispt.flatten())        #sorted indexing in the flattened array
    ret = (n*n - n)/2 #number of relevant items, others are all inf
    #pos2 = [[i/n, i%n] for i in ix[:ret]] # stores the sorted indices
    #if three or more lines intersect at a point, then I am not sure how the above ordering would work. This needs to be checked 

    pos = np.unravel_index(ix[:ret], np.shape(ispt))
    pos = np.column_stack((pos[0], pos[1]))

    
    numIspt = len(pos) #storing the number of intersection points
    
    sigma = np.ones((n-1, numIspt +1) ) * sys.maxint#the number of possible permutations is 1+ the number of intersection points
    sigma[:,0] = 1+np.argsort(-1*v[1:]) #we want to sort v into descending order, so we are sorting -v in ascending order and storing the indexes
    
    A = np.ones((C, numIspt +1) ) * sys.maxint
    G = np.ones((C, numIspt +1) ) * sys.maxint
    B = np.ones((n-1, numIspt +1) ) * sys.maxint
    
    Bcount = -1 #number of elements in current B vector
    
    A[:,0] = sigma[0:C,0]
    G[:,0] = sigma[0:C,0]
    
    for l in range(1,numIspt+1): 
        sigma[:,l] = sigma[:,l-1]
        B[:, l] = B[:, l-1]
        
        #this is to ensure that the first coordinate is smaller -not sure if the below line is foolproof   
        if(pos[l-1][0] > pos[l-1][1]):
            pos[l-1][0], pos[l-1][1] = pos[l-1][1], pos[l-1][0]
        #print 'position switched'
        
        if pos[l-1][0] != 0:
            idx1 = np.where(sigma[:,l-1] == pos[l-1][0])
            idx2 = np.where(sigma[:,l-1] == pos[l-1][1])
            #print sigma[idx1, l-1], sigma[idx2, l-1]
            sigma[idx1, l], sigma[idx2, l] =  sigma[idx2, l-1], sigma[idx1, l-1]
            
        else:
            B[Bcount + 1,l] = pos[l-1][1]
            Bcount = Bcount +1
            
        G[:,l] = sigma[0:C, l]
        temp = np.setdiff1d(G[:,l], B[:,l])
        A[0:len(temp), l ] = temp
                
    
    maxRev= 0 #denoting the maximum revenue encountered in the sets till now
    maxRevSet = -1 #denoting which set has the maximum revenue till now
    
    for l in range(numIspt+1):
        objs = A[np.where(A[:, l]< sys.maxint), l].flatten()           
        #objs = np.where(0 < objs)  
    
        rev = calcRev(objs.astype('int'), p, v, prod)
        if rev > maxRev:
            maxRev = rev
            maxRevSet = l
            
    
    optSet =  A[np.where(A[:, maxRevSet]< sys.maxint), maxRevSet].flatten()
    timeTaken = time.time() - st
    #optSet =  np.where(0 < optSet)
    print " " 
    print "Results for Paat's algorithm"
    print "Products in the optimal assortment are", optSet 
    print "Optimal revenue is", maxRev
    print "Time taken for running Paat's algorithm is", timeTaken       
    return maxRev, optSet, timeTaken;       
        
def capAst_static_mnl(prod,C,p,v,meta=None):
  if prod > 1000:
    print '\n\n\n*******PAAT will NOT execute for #products greater than 1000: using capAst_LP instead*********\n\n'
    from capAst_LP import capAst_LP
    rev, maxSet, time = capAst_LP(prod, C, p, v)
  else:
    rev, maxSet, time = capAst_paat0(prod, C, p, v)
    maxSet = set(maxSet.astype(int))
  return rev,maxSet,time


######### STATIC-MNL #################################




######### LP #################################


def capAst_LP(prod, C, p, v, meta = None):
    #v and p are expected to be n+1 and n+1 length lists respectively 
    st = time.time()    

    # Model
    m = Model("capAst_LP")

    items = range(prod+1)

    item = {}
    for i in items:
      item[i] = m.addVar(lb=0,name='item_'+str(i))

    # The objective is to maximize expected revenue
    m.setObjective(sum(item[i]*p[i] for i in items), GRB.MAXIMIZE)

    # constraint1
    sum1 = 0
    for i in items:
        sum1 += item[i]
    m.addConstr(sum1 == 1, 'sum2one')

    #constraint2
    sum2 = 0
    for i in items[1:]:
        sum2 += item[i]/v[i]
    m.addConstr(sum2 - item[0]*C/v[0] <= 0, "capacity")

    #constraint3 (many)
    for i in items[1:]:
        m.addConstr(item[i]/v[i] - item[0]/v[0] <= 0, 'order_'+str(i))

    # Solve
    m.setParam(GRB.Param.OutputFlag, 0)
    m.optimize()

    timeTaken = time.time() - st      
    # print " " 
    # print "Results for LP"
    print '\t\tTime taken for running the LP is', timeTaken
    maxRev = 0
    maxRevSet = []
    if m.status == GRB.Status.OPTIMAL:
        maxRev = m.objVal
        # print "Optimal revenue is", maxRev

        itemx = m.getAttr('x', item)
        for i in items[1:]:
            if item[i].x*v[0]/max(v[i],0.00001)/max(item[0].x,0.00001) > 0.0001:
                # print('%s %g' % (i, itemx[i]))
                maxRevSet.append(int(i))

        print "\t\tProducts in the LP optimal assortment are", maxRevSet 

    else:
        print('No solution')

    return maxRev, set(maxRevSet),timeTaken



######### LP #################################






######### ADXOPT #################################



def capAst_adxopt(prod, C, p, v, meta = None):

    st = time.time() 
    # initialize
    b = min(C,prod-C+1) #parameter of adxopt, see Thm 3, Jagabathula 
    items = range(1,prod+1)
    removals = np.zeros(prod+1)
    set_prev = []
    rev_prev = calcRev(set_prev, p, v, prod)

    rev_cal_counter = 0
    while True:
        items_left = [x for x in items if x not in set_prev]
        #Additions
        set_addition = []
        rev_addition = 0
        if len(set_prev) < C:
            for j in items_left:
                if removals[j] <b:
                    candidate_rev = calcRev(sorted(set_prev+[j]),p,v,prod)
                    rev_cal_counter +=1
                    if candidate_rev > rev_addition:
                        rev_addition = candidate_rev
                        set_addition = sorted(set_prev+[j])

        #Deletions
        set_deletion = []
        rev_deletion = 0
        if len(set_prev) >0:
            for idx in range(len(set_prev)):
                candidate_rev = calcRev(sorted(set_prev[:idx]+set_prev[idx+1:]),p,v,prod)
                rev_cal_counter +=1
                if candidate_rev > rev_deletion:
                    rev_deletion = candidate_rev
                    set_deletion = sorted(set_prev[:idx]+set_prev[idx+1:])

        #Substitutions
        set_substitution = []
        rev_substitution = 0
        if len(set_prev) >0:
            for j in items_left:
                if removals[j] <b:
                    for idx in range(len(set_prev)):
                        candidate_rev = calcRev(sorted(set_prev[:idx]+[j]+set_prev[idx+1:]),p,v,prod)
                        rev_cal_counter +=1
                        if candidate_rev > rev_substitution:
                            rev_substitution = candidate_rev
                            set_substitution = sorted(set_prev[:idx]+[j]+set_prev[idx+1:])


        idx_rev_current = np.argmax(np.asarray([rev_addition,rev_deletion,rev_substitution]))
        if idx_rev_current==0:
            rev_current = rev_addition
            set_current = set_addition
        elif idx_rev_current==1:
            rev_current = rev_deletion
            set_current = set_deletion
        else:
            rev_current = rev_substitution
            set_current = set_substitution

        if rev_current <= rev_prev or np.min(removals) >= b:
            rev_current = rev_prev
            set_current = set_prev
            break
        else:
            for j in set(set_prev).difference(set(set_current)):
                removals[j] +=1

            rev_prev = rev_current
            set_prev = set_current

    timeTaken = time.time() - st  

    print "\t\tNumber of times calcRev is called:",rev_cal_counter
    print "\t\tProducts in the adxopt assortment are", set_current 
    print '\t\tTime taken for running adxopt is', timeTaken
    
    return rev_current, set_current, timeTaken
        
        
######### ADXOPT #################################

