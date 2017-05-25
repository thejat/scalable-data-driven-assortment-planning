
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
from capAst_oracle import calcRev
import numpy as np
import time, math



def preprocess(prod, C, p, algo, nEst=10,nCand=40,feasibles = None):
    t0 = time.time()

    if ((algo == 'special_case_LSH')| (algo == 'general_case_LSH')):
        db =  LSHForest(n_estimators= nEst, n_candidates=nCand, n_neighbors=C)   
    else:
        db =  NearestNeighbors(n_neighbors=C, metric='cosine', algorithm='brute')   
 
    if ((algo == 'special_case_LSH') | (algo=='special_case_exact')):
        U = np.eye(prod)
        normConst = np.sqrt(1+np.max(p)**2)
        ptsTemp = np.concatenate((U*np.array(p[1:]),U), axis=1)*1/normConst
        # print ptsTemp,ptsTemp.shape,1.0/normConst
    else:
        normConst = 1
        print "tbd"
        ptsTemp = np.zeros((prod,2*prod)) #replace!

    #MIPS to NN transformation of all points
    lastCol = np.linalg.norm(ptsTemp, axis=1)**2
    lastCol = np.sqrt(1-lastCol)
    pts =  np.concatenate((ptsTemp, lastCol.reshape((prod,1))), axis =1)     
    db.fit(pts) 
    build_time = time.time() - t0
    print "LSHF/NN index build time: ", build_time   
    return db, build_time, normConst   
    

def assortX(prod, C, p, v,  eps, algo=None, db=None, normConst=None):
    
    st  = time.time()
    L   = 0 #L is the lower bound of the search space
    U   = max(p) #Scalar here

    while (U - L) > eps:
        K = (U+L)/2
        maxPseudoRev, maxSet= get_nn_set(v,p,K,prod,C,db,normConst,algo)
        if (maxPseudoRev/v[0]) >= K:
            L = K
        else:
            U = K
                    
    maxRev =  calcRev(maxSet, p, v,prod)
    timeTaken = time.time() - st
    
    return maxRev, maxSet, timeTaken
   

def get_nn_set(v,p,K, prod, C, db, normConst,algo):
    vTemp = np.concatenate((v[1:], -K*v[1:]))
    query = np.concatenate((vTemp, [0])) #appending extra coordinate as recommended by Simple LSH, no normalization being done

    distList, approx_neighbors  = db.kneighbors(query.reshape(1,-1),return_distance=True)

    print distList,approx_neighbors

    # if ((algo == 'special_case_LSH') | (algo=='special_case_exact')):    
    real_neighbours = (distList<1) #consider points only whose dot product with v is strictly positive
    real_dist = np.linalg.norm(query)*(1-distList)[0] 
    real_dist=real_dist * normConst

    nn_set = approx_neighbors[0][real_neighbours[0]] + 1 # + 1 is done to identify the product as the indexing would have started from 0
    try:
        nn_set = list(nn_set.astype(int)) #replace
    except:
        nn_set = nn_set

    return sum(real_dist[real_neighbours[0]]), nn_set      
  

# Wrappers
# Assort-Exact-special
def capAst_AssortExact(prod,C,p,v,meta):
  return assortX(prod, C, p, v, 
    meta['eps'],  
    algo = 'special_case_exact',
    db=meta['db_exact'], 
    normConst=meta['normConst'])
# Assort-LSH-special
def capAst_AssortLSH(prod,C,p,v,meta):
  return assortX(prod, C, p, v,
    meta['eps'],  
    algo = 'special_case_LSH', 
    db =meta['db_LSH'], 
    normConst=meta['normConst'])
# Assort-Exact-general
def genAst_AssortExact(prod,C,p,v,meta):
  return assortX(prod, C, p, v, 
    meta['eps'],  
    algo = 'general_case_exact',
    db=meta['db_exact'], 
    normConst=meta['normConst'])
# Assort-LSH-general
def genAst_AssortLSH(prod,C,p,v,meta):
  return assortX(prod, C, p, v,
    meta['eps'],  
    algo = 'general_case_LSH', 
    db =meta['db_LSH'], 
    normConst=meta['normConst'])