# -*- coding: utf-8 -*-
"""
Created on Tue Feb 02 15:12:38 2016

@author: Deeksha
"""

from itertools import combinations
import sys
import numpy
import math
import cylsh
import time
from cylsh import LSH


#epsilon is the tolerance parameter
#pMax is the maximum price i.e. the largest element of the list p. It will be obtained from the pre-processing step. 
def capAst_LSH(prod, C, p, v, eps, R_lsh, p_lsh):   
    
    #initialisation    
    KList, astList, dbList = preprocessing_NNapprox(p, prod, C, R_lsh, p_lsh, eps)
    st  =time.time()
    mismatch_count = 0
    loop_count = 0
    L = 0 #L is the lower bound of the search space
    U = max(p)
    while (U - L) >eps:
        K = (U+L)/2
        #this is for exhaustive search
        #maxPseudoRev_exh, maxSet_exh = calcNN_exhaustive(v, p, K, prod, C ) #replace calcNN_exhaustive with the function use to calculate the maximum inner product
        
        #this is for approximate search
        maxPseudoRev, maxSet= calcNN_approx(v,p,K, prod, C, KList, dbList, astList)
        #if  (maxSet != maxSet_exh):
        #    mismatch_count = mismatch_count +1
        if (maxPseudoRev/v[0]) >= K:
            L = K
        else:
            U= K
            
        loop_count = loop_count +1  
        
    maxRev =  calcRev(maxSet, p, v,prod)
    timeTaken = time.time() - st
    
    print " "
    print "Results for new algorithm"
    #print 'Percentage of mismatch loops is',   mismatch_count/ loop_count     
    print 'Revenue maximising set is ', maxSet
    print 'Revenue for this set is', maxRev
    print 'Time taken for running the new algorithm is', timeTaken
    
    return maxRev, maxSet
            

def calcNN_approx(v,p,K, prod, C, KList, dbList, astList):
    idx = numpy.where(numpy.around(KList, decimals =10) == numpy.around(K, decimals =10))
    ansDict = dbList[idx[0][0]].query(v[1:]) #idx[0][0] to extract the  exact value of the index
    NNs =  ansDict['neighbors']
    ipList = numpy.zeros(len(ansDict['neighbors'])) #list of inner products
    ctr = 0
    for i in NNs:
        ast = astList[i]
        for j in ast:
            ipList[ctr] = ipList[ctr] + v[j]*(p[j] - K)
        ctr = ctr +1    
    #maxIp = max(ipList)
    maxIpIdx = numpy.where(ipList== max(ipList))  
    topNNidx = NNs[maxIpIdx] #storing the element having the highest inner product
    NN= astList[topNNidx[0]] #type of the variable topNNidx is array, thus take the first element
    #ans = (ansDict['neighbors'])[0]    
    #NN = astList[ans]
#    pseudoRev = 0
#    for i in range(len(NN)):
#            pseudoRev = pseudoRev + v[NN[i]]*(p[NN[i]] - K)
    return max(ipList), NN
    
    
    
    
def preprocessing_NNapprox(p, prod, C, R, p_lsh, eps):    
    astList = list(combinations(range(1,prod+1), 1))
    for i in range(2, C+1):
        astList = astList + list(combinations(range(1,prod+1), i))
    
    
    ctr = numpy.ceil(math.log( max(p)/eps, 2))
    KList = numpy.linspace(0, max(p), math.pow(2, ctr)+1 )
    dbList =  [ cylsh.LSH() for i in range(len(KList))]   
    
    #for saving the datasets, for debugging purpose
    datasetList = numpy.zeros((len(astList), prod, len(KList)))   
    
    for K in range(len(KList)):
        dataset = numpy.zeros((len(astList), prod))
        i = 0
        for ast in astList:
            for j in range(len(ast)):
                dataset[i][ast[j]-1] = p[ast[j]] - KList[K]
            i = i+1
        sample_query = dataset[1:3, :]
        # Create LSH Database
        optim_param = LSH.compute_optimal_parameters(R, p_lsh, dataset, sample_query, 22020096)
        dbList[K] = LSH.init_manually(optim_param, dataset)
        datasetList[:,:,K] = dataset
        
        
    return KList, astList, dbList #, datasetList    
    
    
 #For generating the dataset for a particular K   
def genDataSet(p, K, prod, C):   
    astList = list(combinations(range(1,prod+1), 1))
    for i in range(2, C+1):
        astList = astList + list(combinations(range(1,prod+1), i))
    dataset = numpy.zeros((len(astList), prod))
    i = 0
    for ast in astList:
        for j in range(len(ast)):
            dataset[i][ast[j]-1] = p[ast[j]] - K
        i = i+1
    return dataset    
   
   
    



def calcNN_exhaustive(v,p, K, prod, C):
    if len(p)==prod:
        p =  numpy.insert(p,0,0)#making p a n+1 length list by inserting a 0 in the beginning
    maxPseudoRev = -1*sys.maxint
    maxSet = -1    
    astList = list(combinations(range(1,prod+1), 1))
    for i in range(2, C+1):
        astList = astList + list(combinations(range(1,prod+1), i))
    for ast in astList:
        pseudoRev = 0
        for i in range(len(ast)):
            pseudoRev = pseudoRev + v[ast[i]]*(p[ast[i]] - K)
        if  pseudoRev > maxPseudoRev:
            maxPseudoRev = pseudoRev
            maxSet = ast
    return (maxPseudoRev, maxSet)       
        
        
def calcRev(ast, p, v, prod):
#v and p are expected to be n+1 and n length lists respectively 
    if len(p)==prod:    
        p =  numpy.insert(p,0,0)   #making p a n+1 length list by inserting a 0 in the beginning
    num = 0
    den = v[0]
    for s in range(len(ast)):
        num  = num + p[ast[s]]*v[ast[s]]
        den  = den + v[ast[s]]
    rev = num/den
    return rev
        
        
        
    
    