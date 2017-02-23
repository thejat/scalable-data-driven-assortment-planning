# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 21:21:38 2016

@author: Deeksha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 12:06:56 2016

@author: Deeksha
"""

import numpy as np
#from capAst_approx import capAst_LSH
#from capAst_oracle import capAst_oracle
from capAst_scikitLSH import capAst_NNalgo, preprocess
from capAst_paat import capAst_paat
import time
import os
   
 #match answers with the test API
   #dataset = genDataSet(p, K, prod, C)
   #queryset = numpy.reshape(v[1:], (1, prod))
   #test_approxNN('mips', dataset, queryset, R_lsh, p_lsh)
   
    

#parameters required
#prod = 3000 #number of products
C= 10   #capacity of assortment
price_range = 100; #denotes highest possible price of a product
eps = 0.1 #tolerance
N = 50 #number of times Monte Carlo simulation will run
repPV = N #number of times a p, v pair is repeated

#skLearn LSH parameters
#nCand = 100
nEst = 50

#nEstList = [20,40,60,80]
nCandList= [50,100,150,200]
#nCandList= [20]#[50,100]

#prodList = np.array([20,40,60,80] + list(np.arange(100,1000,200)) + [1000,2000])
#prodList = [20,   40,   60,   80,  100]
prodList = [20 ,50, 100, 200, 400, 600, 800, 1000]
#prodList = [300,  500,  700,  900, 1000, 2000]


saveFolder = 'algoStats/stats_pmax_' + str(price_range)+'_C_'+str(C) 
if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

revPctErrExactNN = np.zeros((np.shape(prodList)[0], N))
setOlpExactNN = np.zeros((np.shape(prodList)[0],  N))
corrSetExactNN = np.zeros((np.shape(prodList)[0],  N))

revPctErrLSHNN = np.zeros((np.shape(prodList)[0], np.shape(nCandList)[0], N))
setOlpLSHNN = np.zeros((np.shape(prodList)[0], np.shape(nCandList)[0], N))
corrSetLSHNN = np.zeros((np.shape(prodList)[0], np.shape(nCandList)[0], N))

revExactNN = np.zeros((np.shape(prodList)[0], N))
revLSHNN = np.zeros((np.shape(prodList)[0], np.shape(nCandList)[0], N))
revPaat = np.zeros((np.shape(prodList)[0], N))

timeExactNN = np.zeros((np.shape(prodList)[0], N))
timeLSHNN = np.zeros((np.shape(prodList)[0],  np.shape(nCandList)[0], N))
timePaat = np.zeros((np.shape(prodList)[0], N))

corrSetExactNN_mean, setOlpExactNN_mean,  revPctErrExactNN_max, revPctErrExactNN_mean, revPctErrExactNN_std, timeExactNN_mean, timeExactNN_std, timePaat_mean, timePaat_std =  (np.zeros( np.shape(prodList)[0]) for i in range(9))
corrSetLSHNN_mean, setOlpLSHNN_mean, revPctErrLSHNN_max, revPctErrLSHNN_mean, revPctErrLSHNN_std, timeLSHNN_mean, timeLSHNN_std =  (np.zeros(( np.shape(prodList)[0], np.shape(nCandList)[0])) for i in range(7))



badError = 0

i = 0
for prod in prodList:
    
    #building skLSH and exact NN models
    t0 = time.time()
    t = 0
    while(t<N):
        #print 'Iteration number is ', str(t)
        #generating the price and customer preference vector
        p = np.random.uniform(0,price_range, prod) # create an array of n+1 uniform random numbers between 0 and price_range  
        p = np.around(p, decimals =2)
        p = np.insert(p,0,0) #inserting 0 as the first element to denote the price of the no purchase option
        #v = np.random.rand(prod+1) #v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
         
        dbListskLSH= []
        for nCand in nCandList:
            KList, dbListskLSHTemp, build_time_lshf, normConstList = preprocess(prod, C, p,  eps,  'skLsh', nEst=nEst,nCand=nCand)
            dbListskLSH.append(dbListskLSHTemp)
            
        KList, dbListExact, build_time_exactNN, normConstList = preprocess(prod, C, p,  eps,  'exactNN')    
        
        #KList and normConstList don't depend on the NN algorithm and parameters like nEst and nCand        
        
        for s in range(repPV):
            v = np.random.rand(prod+1) #v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
            revExactNN[i, t], maxSetExactNN, timeExactNN[i, t] = capAst_NNalgo(prod, C, p, v,  eps,  algo = 'exactNN',  preprocessed =True,KList=KList, dbList =dbListExact, normConstList=normConstList)
            #revExactNN[i, t], maxSet, timeExactNN[i, t] = capAst_NNalgo(prod, C, p, v,  eps,  algo = 'exactNN',  preprocessed =False)            
            revPaat[i, t], maxSetPaat, timePaat[i, t] = capAst_paat(prod, C, p, v)
            maxSetPaat = set(maxSetPaat.astype(int))
            
            setOlpExactNN[i,t]  = len(maxSetPaat.intersection(maxSetExactNN))
            corrSetExactNN[i,t] =   int(setOlpExactNN[i,t]==  len(maxSetPaat))
            setOlpExactNN[i,t] = setOlpExactNN[i,t]/len(maxSetPaat) #to normalize
            
            if(revPaat[i, t] - revExactNN[i, t] > eps ):
                badError = badError +1
    
            j = 0
            for nCand in nCandList:
                 revLSHNN[i,j,t], maxSetLSHNN, timeLSHNN[i, j,t] = capAst_NNalgo(prod, C, p, v,  eps,  algo = 'skLSH', nEst =nEst, nCand =nCand , preprocessed =True,KList=KList, dbList =dbListskLSH[j], normConstList=normConstList) 
                 #revLSHNN[i,j,t], maxSet, timeLSHNN[i, j,t] = capAst_NNalgo(prod, C, p, v,  eps,  algo = 'skLSH', nEst =nEst, nCand =nCand , preprocessed =False) 
                 revPctErrLSHNN[i,j,t] = (revPaat[i, t] - revLSHNN[i,j,t])/revPaat[i, t]
                 setOlpLSHNN[i,j,t]  = len(maxSetPaat.intersection(maxSetLSHNN))
                 corrSetLSHNN[i,j,t] =   int(setOlpLSHNN[i,j,t] ==  len(maxSetPaat))        
                 setOlpLSHNN[i,j,t] = setOlpLSHNN[i,j,t]/len(maxSetPaat)           
                 j = j +1
            t = t+1    
            print 'Iteration number is ', str(t)
           
    revPctErrExactNN[i] = (revPaat[i,:] - revExactNN[i,:])/revPaat[i,:] 
    revPctErrExactNN_mean[i] = np.mean(revPctErrExactNN[i,:])
    revPctErrExactNN_std[i] = np.std(revPctErrExactNN[i,:]) 
    revPctErrExactNN_max[i] = np.max(revPctErrExactNN[i,:])
    corrSetExactNN_mean[i] = np.mean(corrSetExactNN[i,:])
    setOlpExactNN_mean[i] = np.mean(setOlpExactNN[i,:])
    
    revPctErrLSHNN_mean[i,:] = np.mean(revPctErrLSHNN[i,:,:], axis = 1)    
    revPctErrLSHNN_std[i,:] = np.std(revPctErrLSHNN[i,:,:], axis =1) 
    revPctErrLSHNN_max[i,:] = np.max(revPctErrLSHNN[i,:,:], axis = 1)
    corrSetLSHNN_mean[i,:] = np.mean(corrSetLSHNN[i,:,:], axis = 1)
    setOlpLSHNN_mean[i,:] = np.mean(setOlpLSHNN[i,:,:], axis = 1)
          
    timeExactNN_mean[i] = np.mean(timeExactNN[i,:])
    timeExactNN_std[i] = np.std(timeExactNN[i,:]) 
    timePaat_mean[i] = np.mean(timePaat[i,:])
    timePaat_std[i] = np.std(timePaat[i,:])
    
    timeLSHNN_mean[i,:] = np.mean(timeLSHNN[i,:,:], axis = 1)    
    timeLSHNN_std[i,:] = np.std(timeLSHNN[i,:,:], axis =1) 

    print 'Calculations done for number of products', prod  
    np.savez(saveFolder + '/stats_prod_'+ str(prod), setOlpExactNN=setOlpExactNN[i], corrSetExactNN=corrSetExactNN[i],setOlpExactNN_mean=setOlpExactNN_mean[i], corrSetExactNN_mean=corrSetExactNN_mean[i], setOlpLSHNN=setOlpLSHNN[i], corrSetLSHNN=corrSetLSHNN[i],setOlpLSHNN_mean=setOlpLSHNN_mean[i], corrSetLSHNN_mean=corrSetLSHNN_mean[i], revPctErrExactNN = revPctErrExactNN[i], revPctErrExactNN_mean = revPctErrExactNN_mean[i], revPctErrExactNN_std = revPctErrExactNN_std[i], revPctErrLSHNN_mean = revPctErrLSHNN_mean[i],  revPctErrLSHNN_std = revPctErrLSHNN_std[i]  , timeExactNN_mean = timeExactNN_mean[i] , timeExactNN_std =timeExactNN_std[i], timePaat_mean =timePaat_mean[i], timePaat_std = timePaat_std[i], timeLSHNN_mean = timeLSHNN_mean[i], timeLSHNN_std = timeLSHNN_std[i], revPctErrLSHNN_max=revPctErrLSHNN_max[i], revPctErrExactNN_max=revPctErrExactNN_max[i] )
    print 'Time taken to run is', time.time() - t0    
    i = i +1

np.savez(saveFolder + '/allStats_pmax_' + str(price_range)+'_C_'+str(C) , setOlpExactNN_mean=setOlpExactNN_mean, corrSetExactNN_mean=corrSetExactNN_mean, setOlpLSHNN_mean=setOlpLSHNN_mean, corrSetLSHNN_mean=corrSetLSHNN_mean,  revPctErrExactNN_mean = revPctErrExactNN_mean, revPctErrExactNN_std = revPctErrExactNN_std, revPctErrLSHNN_mean = revPctErrLSHNN_mean,  revPctErrLSHNN_std = revPctErrLSHNN_std  , timeExactNN_mean = timeExactNN_mean , timeExactNN_std =timeExactNN_std, timePaat_mean =timePaat_mean, timePaat_std = timePaat_std, timeLSHNN_mean = timeLSHNN_mean, timeLSHNN_std = timeLSHNN_std )


#revPctErrExactNN = (revPaat - revExactNN)/revPaat    
#revPctErrExactNN_mean = np.mean(revPctErrExactNN, axis=1)    
#revPctErrExactNN_std = np.std(revPctErrExactNN, axis=1) 
#
#revPctErrLSHNN_mean = np.mean(revPctErrLSHNN, axis=2)    
#revPctErrLSHNN_std = np.std(revPctErrLSHNN, axis=2)  
#
#timeExactNN_mean = np.mean(timeExactNN, axis=1)
#timePaat_mean = np.mean(timePaat, axis=1)
#timeLSHNN_mean = np.mean(timeLSHNN, axis=2)
#timeExactNN_std = np.std(timeExactNN, axis=1)
#timeLSHNN_std = np.std(timeLSHNN, axis=1)
#timePaat_std = np.std(timePaat, axis=1)
#    
    

print 'Bad error number is ',   badError 