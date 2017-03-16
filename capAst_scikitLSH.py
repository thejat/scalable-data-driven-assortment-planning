# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:50:40 2016

@author: Deeksha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 02 15:12:38 2016

@author: Deeksha
"""

from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
#from capAst_approx import calcNN_exhaustive
from capAst_oracle import calcRev
import numpy as np
import time



import sys
import math





def capAst_NNalgo(prod, C, p, v,  eps,  algo = 'exactNN', nEst =10, nCand =40 , preprocessed =False, KList=None, dbList=None, normConstList=None):
    #nCan is the number of candidates - a parameter required for LSH Forest   
    
    if(not(preprocessed)):
#        if (algo == 'skLSH'):
#            KList, dbList, build_time_lshf, normConstList = preprocess(prod, C, p,  eps,  'skLSH', nEst,nCand)
#        elif (algo == 'exactNN'):
#            KList, dbList, build_time_lshf, normConstList = preprocess(prod, C, p,  eps,  'exactNN')
#        elif (algo == 'skLSH_singleLSH'):
#            dbList, build_time_lshf, normConstList = preprocess(prod, C, p,  eps,  'skLSH_singleLSH', nEst,nCand)       
        KList, dbList, build_time_lshf, normConstList = preprocess(prod, C, p,  eps,  algo, nEst,nCand)

       
    #print 'KList is ', KList
    
    st  =time.time()
    #mismatch_count = 0
    loop_count = 0
    L = 0 #L is the lower bound of the search space
    U = max(p)
    while (U - L) >eps:
        K = (U+L)/2
        #this is for exhaustive search
        #maxPseudoRev_exh, maxSet_exh = calcNN_exhaustive(v, p, K, prod, C ) #replace calcNN_exhaustive with the function use to calculate the maximum inner product
        
        #this is for approximate search
        maxPseudoRev, maxSet= calcNN(v,p,K, prod, C, KList, dbList, normConstList,algo)
        #if  (maxSet != maxSet_exh):
        #    mismatch_count = mismatch_count +1
        if (maxPseudoRev/v[0]) >= K:
            L = K
        else:
            U= K
            
        loop_count = loop_count +1  
        
    maxRev =  calcRev(maxSet, p, v,prod)
    timeTaken = time.time() - st
#    
#    print " "
#    print "Results for new algorithm"
#    #print 'Percentage of mismatch loops is',   mismatch_count/ loop_count     
#    print 'Revenue maximising set is ', maxSet
#    print 'Revenue for this set is', maxRev
#    print 'Time taken for running the ', algo, ' based algorithm is', timeTaken
#    
    return maxRev, maxSet, timeTaken
   
        



def preprocess(prod, C, p,  eps,  algo, nEst=10,nCand=40):
    t0 = time.time()  
    U = np.eye(prod)     
    ctr = np.ceil(math.log( max(p)/eps, 2))
    KList = np.linspace(0, max(p), math.pow(2, ctr) +1) 
    KList = np.delete(KList, np.shape(KList)[0] -1) # K will never actually take the value p_max
    
    
    
    if(algo=='skLSH'):
        dbList =  [ LSHForest(n_estimators= nEst , n_candidates=nCand, n_neighbors=C) for i in range(len(KList))]   
    elif(algo == 'exactNN'):
        dbList =  [ NearestNeighbors(n_neighbors=C, metric='cosine', algorithm='brute')  for i in range(len(KList))]   
    elif (algo == 'skLSH_singleLSH'):
        dbList =  [ LSHForest(n_estimators= nEst , n_candidates=nCand, n_neighbors=C) ]   
        KList = [0]
    elif (algo == 'exactNN_single'):
        dbList =  [ NearestNeighbors(n_neighbors=C, metric='cosine', algorithm='brute')  ]   
        KList = [0]
    normConstList = np.zeros(np.shape(KList)[0])
    #for saving the datasets, for debugging purpose
    #datasetList = numpy.zeros((len(astList), prod, len(KList)))     
    for K in range(len(KList)):        
         if ((algo == 'skLSH_singleLSH') | (algo=='exactNN_single')):
            normConstList[K] = np.sqrt(1+np.max(p[1:])**2)
            ptsTemp = np.concatenate((U*np.array(p[1:]),U),  axis=1)/normConstList[K]
         else:
            #All the points which have negative p(i) - K  will be rounded to 0 as they will have negative dot product with v 
            #and hence will not be considered in the top C neighbour calculation
            normConstList[K] = np.max(np.array(p[1:])-KList[K])
            ptsTemp = U* np.maximum((np.array(p[1:])-KList[K])/normConstList[K],0) #(np.array(p[1:])-K)
            #ptsTemp = U* ((np.array(p[1:])-KList[K])/divideBy)      
            #append extra column to all points to make it have norm 1
         lastCol = np.linalg.norm(ptsTemp, axis=1)**2
         lastCol = np.sqrt(1-lastCol)
         pts =  np.concatenate((ptsTemp, lastCol.reshape((prod,1))), axis =1)     
         dbList[K].fit(pts) 
    build_time_lshf = time.time() - t0
    print "LSHF index build time: ", build_time_lshf   
    return KList, dbList, build_time_lshf, normConstList #, datasetList   
    
   
    


def calcNN(v,p,K, prod, C, KList, dbList, normConstList,algo):
    if ((algo == 'skLSH_singleLSH') | (algo=='exactNN_single')):
        idx = [[0]]
        vTemp = np.concatenate((v[1:], -K*v[1:]))
    else:
        prec = 10**(-5)
        closElem = KList.flat[np.abs(KList - K).argmin()]
        if(np.abs(closElem - K) < prec):
            idx = np.where(KList == closElem)
        else:
            print 'No element close to K'
        vTemp = v[1:]
    query = np.concatenate((vTemp, [0])) #appending extra coordinate as recommended by Simple LSH
    #print 'K is ', K
    #print 'value of index of k is ', idx[0][0]
    distList, approx_neighbors_lshf  = dbList[idx[0][0]].kneighbors(query.reshape(1,-1),return_distance=True)    
    real_neighbours = (distList<1) #consider points only whose dot product with v is strictly positive
    real_dist = np.linalg.norm(query)*(1-distList)[0] 
    real_dist=real_dist * normConstList[idx[0][0]]

    return sum(real_dist[real_neighbours[0]]), approx_neighbors_lshf[0][real_neighbours[0]] + 1 # + 1 is done to identify the product as the indexing would have started from 0
        
  


    