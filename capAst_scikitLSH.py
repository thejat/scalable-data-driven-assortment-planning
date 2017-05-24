
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
from capAst_oracle import calcRev
import numpy as np
import time, math



def preprocess(prod, C, p,  eps,  algo, nEst=10,nCand=40):
    t0 = time.time()


    if ((algo == 'skLSH_singleLSH') | (algo=='exactNN_single')):
        
        KList = [0]
        if (algo == 'skLSH_singleLSH'):
            dbList =  [LSHForest(n_estimators= nEst, n_candidates=nCand, n_neighbors=C)]   
        elif (algo == 'exactNN_single'):
            dbList =  [NearestNeighbors(n_neighbors=C, metric='cosine', algorithm='brute')]   

    else:

        ctr = np.ceil(math.log( max(p)/eps, 2))
        KList = np.linspace(0, max(p), math.pow(2, ctr) +1) 
        KList = np.delete(KList, np.shape(KList)[0] -1) # K will never actually take the value p_max
        if(algo=='skLSH'):
            dbList =  [ LSHForest(n_estimators= nEst, n_candidates=nCand, n_neighbors=C) for i in range(len(KList))]   
        elif(algo == 'exactNN'):
            dbList =  [ NearestNeighbors(n_neighbors=C, metric='cosine', algorithm='brute')  for i in range(len(KList))]   
    
    normConstList = np.zeros(np.shape(KList)[0])
    U = np.eye(prod)     
    for K in range(len(KList)):
        if ((algo == 'skLSH_singleLSH') | (algo=='exactNN_single')):
            normConstList[K] = np.sqrt(1+np.max(p)**2)
            ptsTemp = np.concatenate((U*np.array(p[1:]),U), axis=1)*1/normConstList[K]
            print ptsTemp,ptsTemp.shape,1.0/normConstList[K]
        else:
            #All the points which have negative p(i) - K  will be rounded to 0 as they will have negative dot product with v 
            #and hence will not be considered in the top C neighbour calculation
            normConstList[K] = np.max(np.array(p)-KList[K])
            ptsTemp = U* np.maximum((np.array(p[1:])-KList[K])/normConstList[K],0)
            #ptsTemp = U* ((np.array(p[1:])-KList[K])/divideBy)      
            #append extra column to all points to make it have norm 1
        lastCol = np.linalg.norm(ptsTemp, axis=1)**2
        lastCol = np.sqrt(1-lastCol)
        pts =  np.concatenate((ptsTemp, lastCol.reshape((prod,1))), axis =1)     
        dbList[K].fit(pts) 
    build_time_lshf = time.time() - t0
    print "LSHF index build time: ", build_time_lshf   
    return KList, dbList, build_time_lshf, normConstList   
    

def capAst_NNalgo(prod, C, p, v,  eps,  
    algo = 'exactNN', KList=None, dbList=None, normConstList=None):
    #nCan is the number of candidates - a parameter required for LSH Forest   
    
    st  = time.time()
    L   = 0 #L is the lower bound of the search space
    U   = max(p)

    while (U - L) > eps:
        K = (U+L)/2
        maxPseudoRev, maxSet= calcNN(v,p,K,prod,C,KList,dbList,normConstList,algo)
        if (maxPseudoRev/v[0]) >= K:
            L = K
        else:
            U = K
                    
    maxRev =  calcRev(maxSet, p, v,prod)
    timeTaken = time.time() - st
    
    return maxRev, maxSet, timeTaken
   

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
        
  



#### general ones

#Preprocessing for Assort-Exact and Assort-LSH
def preprocess_general_assortx(prod,feasibles,p,eps):
  nEst = 100
  nCand = 200
  _, dbListskLSH,_,_ = preprocess_general(prod, feasibles, p,  eps,  'skLSH_singleLSH', nEst=nEst,nCand=nCand)    
  KList, dbListExact, _, normConstList = preprocess_general(prod, feasibles, p,  eps,  'exactNN_single')
  return {'dbListskLSH':dbListskLSH,'dbListExact':dbListExact,'KList':KList,'normConstList':normConstList,'eps':eps,'nEst':nEst,'nCand':nCand}


# Assort-Exact
def capAst_AssortExact_general(prod,feasibles,p,v,meta):
  rev, maxSet, time = capAst_NNalgo_general(prod, C, p, v, 
    meta['eps'],  
    algo = 'exactNN_single',
    preprocessed =True,
    KList=meta['KList'], 
    dbList=meta['dbListExact'], 
    normConstList=meta['normConstList'])
  maxSet = set(maxSet.astype(int))
  return rev,maxSet,time
# Assort-LSH
def capAst_AssortLSH_general(prod,feasibles,p,v,meta):
  rev, maxSet, time = capAst_NNalgo_general(prod, C, p, v,
    meta['eps'],  
    algo = 'skLSH_singleLSH', 
    nEst =meta['nEst'], 
    nCand =meta['nCand'] , 
    preprocessed =True,
    KList=meta['KList'], 
    dbList =meta['dbListskLSH'], 
    normConstList=meta['normConstList']) 
  return rev,maxSet,time

def capAst_NNalgo_general(prod, feasibles, p, v,  eps,  algo = 'exactNN', nEst =10, nCand =40 , KList=None, dbList=None, normConstList=None):
    return NotImplementedError

def preprocess_general(prod, feasibles, p,  eps,  algo, nEst=10,nCand=40):
    return None,None,None,None #NOT IMPLEMENTED