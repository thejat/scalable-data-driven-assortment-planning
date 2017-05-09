import numpy as np
#from capAst_approx import capAst_LSH
#from capAst_oracle import capAst_oracle
from capAst_scikitLSH import capAst_NNalgo, preprocess
from capAst_paat import capAst_paat
import time
import os, random
   
 #match answers with the test API
   #dataset = genDataSet(p, K, prod, C)
   #queryset = numpy.reshape(v[1:], (1, prod))
   #test_approxNN('mips', dataset, queryset, R_lsh, p_lsh)
   
    

#parameters required
#prod = 3000 #number of products
C= 50   #capacity of assortment
price_range = 1000; #denotes highest possible price of a product
eps = 0.1 #tolerance
N = 50 #number of times Monte Carlo simulation will run
repPV = 1 #number of times a p, v pair is repeated
genMethod = 'bppData' #specifies the method of generating price vectors

fname = os.getcwd() + '/billion_price_data/processed_data/usa_2/numProducts_stats.npz'
dateList = np.load(fname)['good_dates']


fileName = os.getcwd() + '/billion_price_data/processed_data/usa_2/prices_'
fileNameList = []
#for day in range(10,31):
#    for month in ['jun', 'jul', 'aug']:
#        chosenDay = str(day) + month + '2009'
for chosenDay in dateList:
    fileNameList.append(fileName+ chosenDay+'.npz')


#skLearn LSH parameters
#nCand = 100
nEst = 100

#nEstList = [20,40,60,80]
nCandList= [50,200,500]
#nCandList= [20]#[50,100]

#prodList = np.array([20,40,60,80] + list(np.arange(100,1000,200)) + [1000,2000])
prodList = [100, 200, 400, 600, 800, 1000, 2000, 5000, 10000,15000,20000]
#prodList = [100,200]


saveFolder = 'algoStats_bpp/stats_pmax_' + str(price_range)+'_C_'+str(C) 
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

def get_price(price_range, prod, genMethod, iterNum = 0, fileNameList=None):
#genMethod can take values 'uniform', 'beta', 'bppData'
    if(genMethod=='uniform'):
        p = np.random.uniform(0,price_range, prod)
    elif(genMethod=='beta'):
        p = price_range * np.random.beta(2,5,prod) 
    elif(genMethod=='bppData'):
        allPrices = np.load(fileNameList[iterNum])['arr_0']
        allValidPrices = allPrices[allPrices < price_range]
        p = random.sample(allValidPrices, prod)
    return p    
        

badError = 0
i = 0
t1= time.time()
for prod in prodList:
    
    #building skLSH and exact NN models
    t0 = time.time()
    t = 0
    while(t<N):
        #print 'Iteration number is ', str(t)
        #generating the price and customer preference vector
    
        #p = np.random.uniform(0,price_range, prod) # create an array of n+1 uniform random numbers between 0 and price_range  
        #p = price_range * np.random.beta(2,5,prod)
        p = get_price(price_range, prod, genMethod,t, fileNameList)
        p = np.around(p, decimals =2)
        p = np.insert(p,0,0) #inserting 0 as the first element to denote the price of the no purchase option
        if(genMethod=='bppData'):
            max_Price = np.max(p)
            if(max_Price > price_range):
                print 'Maximum price exceeds perrmissible price range '
        #v = np.random.rand(prod+1) #v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
         
        dbListskLSH= []
        for nCand in nCandList:
            KList, dbListskLSHTemp, build_time_lshf, normConstList = preprocess(prod, C, p,  eps,  'skLSH_singleLSH', nEst=nEst,nCand=nCand)
            dbListskLSH.append(dbListskLSHTemp)
            
        KList, dbListExact, build_time_exactNN, normConstList = preprocess(prod, C, p,  eps,  'exactNN_single')    
        
        #KList and normConstList don't depend on the NN algorithm and parameters like nEst and nCand        
        
        for s in range(repPV):
            v = np.random.rand(prod+1) #v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
            v = np.around(v, decimals =7)
            
            #Ensure that there are no duplicate entires in v - required for Paat's algorithm
            u, indices = np.unique(v, return_inverse=True)   
            ct = 1
            while(not(len(u)== prod+1)):
                #print len(u)
                extraSize = prod+1 - len(u)
                newEnt = np.random.rand(extraSize)
                newEnt = np.around(newEnt, decimals =2) 
                v= np.concatenate((u,newEnt))
                u, indices = np.unique(v, return_inverse=True)
                ct =ct+1
            print 'Number of times v had to be generated', ct    
                
            while(v[0]==0):
                v[0] = np.random.rand(1)
             
            revExactNN[i, t], maxSetExactNN, timeExactNN[i, t] = capAst_NNalgo(prod, C, p, v,  eps,  algo = 'exactNN_single',  preprocessed =True,KList=KList, dbList =dbListExact, normConstList=normConstList)
            #revExactNN[i, t], maxSet, timeExactNN[i, t] = capAst_NNalgo(prod, C, p, v,  eps,  algo = 'exactNN',  preprocessed =False)            
            if(prod <= 1000):      #for less than 1000 products calculate Paat's optimal sets            
                revPaat[i, t], maxSetPaat, timePaat[i, t] = capAst_paat(prod, C, p, v)
                maxSetPaat = set(maxSetPaat.astype(int))
            else: #for more than 1000 products, it is tricky to run Paat's algorithm, so we use exactNN answers as a proxy for the true answers
                revPaat[i, t]= revExactNN[i, t]
                maxSetPaat = set(maxSetExactNN.astype(int))
                timePaat[i, t] = timeExactNN[i, t]
                
            setOlpExactNN[i,t]  = len(maxSetPaat.intersection(maxSetExactNN))
            corrSetExactNN[i,t] =   int(setOlpExactNN[i,t]==  len(maxSetPaat))
            setOlpExactNN[i,t] = setOlpExactNN[i,t]/len(maxSetPaat) #to normalize
            
            if(revPaat[i, t] - revExactNN[i, t] > eps ):
                badError = badError +1
    
            j = 0
            for nCand in nCandList:
                 revLSHNN[i,j,t], maxSetLSHNN, timeLSHNN[i, j,t] = capAst_NNalgo(prod, C, p, v,  eps,  algo = 'skLSH_singleLSH', nEst =nEst, nCand =nCand , preprocessed =True,KList=KList, dbList =dbListskLSH[j], normConstList=normConstList) 
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
    #np.savez(saveFolder + '/rawData_prod_'+ str(prod),  )    
    print 'Time taken to run is', time.time() - t0    
    i = i +1

np.savez(saveFolder + '/allStats_pmax_' + str(price_range)+'_C_'+str(C) , setOlpExactNN_mean=setOlpExactNN_mean, corrSetExactNN_mean=corrSetExactNN_mean, setOlpLSHNN_mean=setOlpLSHNN_mean, corrSetLSHNN_mean=corrSetLSHNN_mean,  revPctErrExactNN_mean = revPctErrExactNN_mean, revPctErrExactNN_max = revPctErrExactNN_max, revPctErrExactNN_std = revPctErrExactNN_std, revPctErrLSHNN_mean = revPctErrLSHNN_mean,  revPctErrLSHNN_std = revPctErrLSHNN_std  , revPctErrLSHNN_max = revPctErrLSHNN_max  , timeExactNN_mean = timeExactNN_mean , timeExactNN_std =timeExactNN_std, timePaat_mean =timePaat_mean, timePaat_std = timePaat_std, timeLSHNN_mean = timeLSHNN_mean, timeLSHNN_std = timeLSHNN_std )

print 'Bad error number is ',   badError 
print 'Time taken is', time.time()  - t1