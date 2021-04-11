
from sklearn.neighbors import LSHForest
from sklearn.neighbors import NearestNeighbors
from competing_algos import calcRev
import numpy as np
import time, math
import random
import pickle



def preprocess(prod, C, p, algo, nEst=10,nCand=40,feasibles = None):
    t0 = time.time()

    if algo == 'special_case_LSH':
        print "\tLSH DB Special init..."
        db =  LSHForest(n_estimators= nEst, n_candidates=nCand, n_neighbors=C, min_hash_match=3)
    elif algo=='general_case_LSH':
        print "\tLSH DB General init..."
        db =  LSHForest(n_estimators= nEst, n_candidates=nCand, n_neighbors=1, min_hash_match=3)
    elif algo=="Special_case_BZ":
        print "\tBZ DB Special init..."
        db = LSHForest(n_estimators= nEst, n_candidates=nCand, n_neighbors=C, min_hash_match=3)
    elif algo=='general_case_BZ':
        print "\tLSH DB General init..."
        db =  LSHForest(n_estimators= nEst, n_candidates=nCand, n_neighbors=1, min_hash_match=3)
    elif algo=='special_case_exact':
        print "\tExact DB Special init..."
        db =  NearestNeighbors(n_neighbors=C, metric='cosine', algorithm='brute') 
    else:
        print "\tExact DB General init..."
        db =  NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')   
 
    if ((algo == 'special_case_LSH') | (algo=='special_case_exact') | (algo=='special_case_BZ')):
        U = np.eye(prod)
        normConst = np.sqrt(2+np.max(p)**2)
        ptsTemp = np.concatenate((U*np.array(p[1:]),U), axis=1)*1.0/normConst
        # print ptsTemp,ptsTemp.shape,1.0/normConst
        feasibles = [0 for i in range(ptsTemp.shape[0])] #dummy
    else:
        normConst = C*np.sqrt(1+np.max(p)**2)
        ptsTemp = np.zeros((len(feasibles),2*prod))
        for idx,feasible in enumerate(feasibles):
            ptsTemp[idx] = np.concatenate((np.array(p[1:])*feasible,feasible))*1.0/normConst

    #MIPS to NN transformation of all points
    lastCol = np.linalg.norm(ptsTemp, axis=1)**2
    lastCol = np.sqrt(1-lastCol)
    pts =  np.concatenate((ptsTemp, lastCol.reshape((len(feasibles),1))), axis =1)     
    
    # for e,fe in enumerate(feasibles):
    #   print e,np.linalg.norm(p[1:]*feasibles[e]/normConst),np.linalg.norm(pts[e])
    # NearestNeighbors(n_estimators= nEst, n_candidates=nCand, n_neighbors=C)


    db.fit(pts) 

    build_time = time.time() - t0
    print "\t\tIndex build time: ", build_time   
    
    return db, build_time, normConst#,pts   
    

def assortX(prod, C, p, v,  eps, algo=None, db=None, normConst=None,feasibles=None):
    
    st  = time.time()
    L   = 0 #L is the lower bound of the search space
    U   = max(p) #Scalar here
    count = 0
    queryTimeLog = 0
    while (U - L) > eps:
        K = (U+L)/2
        maxPseudoRev, maxSet,queryTimeLog= get_nn_set(v,p,K,prod,C,db,normConst,algo,feasibles,queryTimeLog)
        if (maxPseudoRev/v[0]) >= K:
            L = K
            # print "going left at count ",count
        else:
            U = K
            # print "going right at count",count
        count +=1

                    
    maxRev =  calcRev(maxSet, p, v,prod)
    timeTaken = time.time() - st
    return maxRev, maxSet, timeTaken, queryTimeLog
   

def get_nn_set(v,p,K, prod, C, db, normConst,algo,feasibles=None,queryTimeLog=0):
    vTemp = np.concatenate((v[1:], -K*v[1:]))
    query = np.concatenate((vTemp, [0])) #appending extra coordinate as recommended by Simple LSH, no normalization being done

    # print "query",query
    # print "query reshaped", query.reshape(1,-1)

    t_before = time.time()
    distList, approx_neighbors  = db.kneighbors(query.reshape(1,-1),return_distance=True)
    queryTimeLog += time.time() - t_before

    # print "distList",distList
    # print distList<1
    # print 1-distList[0]
    # print "approx neigh", approx_neighbors

    if ((algo == 'special_case_LSH') | (algo=='special_case_exact') | (algo=='special_case_BZ')):
        real_neighbours = (distList<1) #consider points only whose dot product with v is strictly positive    
        real_dist = np.linalg.norm(query)*(1-distList)[0] 
        real_dist = real_dist * normConst

        nn_set = approx_neighbors[0][real_neighbours[0]] + 1 # + 1 is done to identify the product as the indexing would have started from 0

        pseudoRev = sum(real_dist[real_neighbours[0]])

    else:
        nn_set = []
        # print 'approx nbhrs', approx_neighbors[0][0]
        # print feasibles[0]
        for idx in range(len(feasibles[0])):
            if feasibles[approx_neighbors[0][0]][idx]==1:
                nn_set.append(idx+1)

        pseudoRev = np.linalg.norm(query)*(1-distList)*normConst 
        # pseudoRev = calcRev(nn_set, p, v, prod)

    try:
        nn_set = list(nn_set.astype(int)) #replace
    except:
        nn_set = nn_set

    return pseudoRev, nn_set,queryTimeLog      
  

# Wrappers
# Assort-Exact-special
def capAst_AssortExactOLD(prod,C,p,v,meta):
  maxRev, maxSet, timeTaken, queryTimeLog = assortX(prod, C, p, v, 
    meta['eps'],  
    algo = 'special_case_exact',
    db=meta['db_exact'], 
    normConst=meta['normConst'])
  print "\t\tAssortExact set:",maxSet
  print "\t\tAssortExact cumulative querytime:",queryTimeLog
  return maxRev, maxSet, timeTaken
# Assort-LSH-special
def capAst_AssortLSH(prod,C,p,v,meta):
  maxRev, maxSet, timeTaken, queryTimeLog = assortX(prod, C, p, v,
    meta['eps'],  
    algo = 'special_case_LSH', 
    db =meta['db_LSH'], 
    normConst=meta['normConst'])
  print "\t\tAssortLSH set:",maxSet
  print "\t\tAssortLSH cumulative querytime:",queryTimeLog
  return maxRev, maxSet, timeTaken
# Assort-Exact-general
def genAst_AssortExact(prod,C,p,v,meta):
  maxRev, maxSet, timeTaken, queryTimeLog = assortX(prod, C, p, v, 
    meta['eps'],  
    algo = 'general_case_exact',
    db=meta['db_exact'], 
    normConst=meta['normConst'],
    feasibles=meta['feasibles'])
  print "\t\tAssortExact-G set:",maxSet
  print "\t\tAssortExact-G cumulative querytime:",queryTimeLog
  return maxRev, maxSet, timeTaken
# Assort-LSH-general
def genAst_AssortLSH(prod,C,p,v,meta):
  maxRev, maxSet, timeTaken, queryTimeLog = assortX(prod, C, p, v,
    meta['eps'],  
    algo = 'general_case_LSH', 
    db =meta['db_LSH'], 
    normConst=meta['normConst'],
    feasibles=meta['feasibles'])
  print "\t\tAssortLSH-G set:",maxSet
  print "\t\tAssortLSH-G cumulative querytime:",queryTimeLog
  return maxRev, maxSet, timeTaken


# Assort-Exact-Linear-Scan
def capAst_AssortExact(prod,C,p,v,meta):

  def createArray(pminusk,v):
    return np.multiply(pminusk,v)

  def linearSearch(p,k,v,C,prod):
    start = time.time()
    maxPseudoRev = 0
    maxSet = []
    bigArray = createArray(p-K,v) 
    candidate_product_idxes = np.argsort(bigArray)[prod+1-C:]
    maxSet = sorted(candidate_product_idxes[bigArray[candidate_product_idxes] > 0])
    maxPseudoRev = sum(bigArray[maxSet])
    return maxPseudoRev,maxSet,time.time()-start


  st  = time.time()
  L   = 0 #L is the lower bound of the search space
  U   = max(p) #Scalar here

  count = 0
  while (U - L) >  meta['eps']:
    K = (U+L)/2  
    maxPseudoRev, maxSet,queryTimeLog = linearSearch(p,K,v,C,prod)
    print "\t\t\tAssortExact querytime:",queryTimeLog, " for K=",K

    if (maxPseudoRev/v[0]) >= K:
      L = K
      # print "going left at count ",count
    else:
      U = K
      # print "going right at count",count
    count +=1
               
  maxRev =  calcRev(maxSet, p, v,prod)
  timeTaken = time.time() - st

  print "\t\tAssortExact Opt Set Size:",len(maxSet)
  print "\t\tAssortExact Opt Set:",maxSet
  print "\t\tAssortExact Opt Rev:",maxRev
  return maxRev, maxSet, timeTaken


#Assort-BZ
def capAst_AssortBZ(prod, C, p, v, meta):
    
    L = 0  # L is the lower bound on the objectiv

    st = time.time()
    queryTimeLog = 0
    count = 0
    
    
    if meta.get('eps', None) is None:
        meta['eps'] = 1e-3

    U = max(p)  # U is the upper bound on the objective
    best_set_revenue = -1
    best_set = []
    L = meta['eps']
    
    # Inititate NBS parameters and define helper functions
    #compstep_prob = meta['default_correct_compstep_probability']
    compstep_prob = 1
    if 'correct_compstep_probability' in meta.keys():
        if meta['correct_compstep_probability'] >= 0.5:
            compstep_prob = meta['correct_compstep_probability']

    
    step_width = 1e-1
    max_iters = 1000
    early_termination_width = 1
    belief_fraction = 0.95


    # Initialize Uniform Distribution
    range_idx = np.arange(L, U, step_width)
    range_dist = np.ones_like(range_idx)
    range_dist = range_dist / np.sum(range_dist)

    range_dist = np.log(range_dist)

    def get_pivot(range_dist):
        exp_dist = np.exp(range_dist)
        alpha = exp_dist.sum() * 0.5

        # Finding the median of the distribution requires
        # adding together many very small numbers, so it's not
        # very stable. In part, we address this by randomly
        # approaching the median from below or above.
        if random.choice([True, False]):
            try:
                return range_idx[exp_dist.cumsum() < alpha][-1]
            except:
                return range_idx[::-1][exp_dist[::-1].cumsum() < alpha][-1]
        else:
            return range_idx[::-1][exp_dist[::-1].cumsum() < alpha][-1]

    def get_belief_interval(range_dist, fraction=belief_fraction):
        exp_dist = np.exp(range_dist)

        epsilon = 0.5 * (1 - fraction)
        epsilon = exp_dist.sum() * epsilon
        if (exp_dist[0] < epsilon):
            left = range_idx[exp_dist.cumsum() < epsilon][-1]
        else:
            left = 0
        right = range_idx[exp_dist.cumsum() > (exp_dist.sum() - epsilon)][0]
        return left, right

    for i in range(max_iters):
        count += 1
        # get Median of Distribution
        median = get_pivot(range_dist)
        # comparision function
        maxPseudoRev, maxSet, queryTimeLog = get_nn_set(v, p, median, prod, C, db = meta['db_BZ'], normConst = meta['normConst'], algo = 'special_case_BZ', feasibles = None, queryTimeLog = 0)
        # Compare Set Revenue with bestSet provided, and replace bestSet if more optimal
        #current_set_revenue = rcm_calc_revenue(maxSet, p, rcm, num_prods)
        current_set_revenue = calcRev(maxSet, p, v, prod)
        if current_set_revenue > best_set_revenue:
            best_set, best_set_revenue = maxSet, current_set_revenue

        if (maxPseudoRev / v[0]) >= median:
            range_dist[range_idx >= median] += np.log(compstep_prob)
            range_dist[range_idx < median] += np.log(1 - compstep_prob)
        else:
            range_dist[range_idx <= median] += np.log(compstep_prob)
            range_dist[range_idx > median] += np.log(1 - compstep_prob)

        # shift all density from lower than best revenue got into upper end
        shift_density_total = np.sum(np.exp(range_dist[range_idx < best_set_revenue]))
        if (shift_density_total > 0):
            range_dist[range_idx < best_set_revenue] = np.log(0)
            range_dist[range_idx >= best_set_revenue] += np.log(
                    shift_density_total / len(range_dist[range_idx >= best_set_revenue]))
        # avoid overflows
        range_dist -= np.max(range_dist)
        belief_start, belief_end = get_belief_interval(range_dist)
 
        if (belief_end - belief_start) <= early_termination_width:
            break

    timeTaken = time.time()-st

    print "\t\tAssortBZ Opt Set Size:",len(best_set)
    print "\t\tAssortBZ Opt Set:",best_set

    return best_set_revenue, best_set, timeTaken 


def genAst_AssortBZ(prod, C, p, v, meta):
    
    
    L = 0  # L is the lower bound on the objectiv

    st = time.time()
    queryTimeLog = 0
    count = 0

    U = max(p)  # U is the upper bound on the objective
    best_set_revenue = -1
    best_set = []
    
    # Inititate NBS parameters and define helper functions
    #compstep_prob = meta['default_correct_compstep_probability']
    compstep_prob = 0.99
    if 'correct_compstep_probability' in meta.keys():
        if meta['correct_compstep_probability'] >= 0.5:
            compstep_prob = meta['correct_compstep_probability']

    step_width = 1e-1
    max_iters = 1000
    early_termination_width = 1
    belief_fraction = 0.95


    # Initialize Uniform Distribution
    range_idx = np.arange(L, U, step_width)
    range_dist = np.ones_like(range_idx)
    range_dist = range_dist / np.sum(range_dist)

    range_dist = np.log(range_dist)

    def get_pivot(range_dist):
        exp_dist = np.exp(range_dist)
        alpha = exp_dist.sum() * 0.5

        # Finding the median of the distribution requires
        # adding together many very small numbers, so it's not
        # very stable. In part, we address this by randomly
        # approaching the median from below or above.
        if random.choice([True, False]):
            try:
                return range_idx[exp_dist.cumsum() < alpha][-1]
            except:
                return range_idx[::-1][exp_dist[::-1].cumsum() < alpha][-1]
        else:
            return range_idx[::-1][exp_dist[::-1].cumsum() < alpha][-1]

    def get_belief_interval(range_dist, fraction=belief_fraction):
        exp_dist = np.exp(range_dist)

        epsilon = 0.5 * (1 - fraction)
        epsilon = exp_dist.sum() * epsilon
        if (exp_dist[0] < epsilon):
            left = range_idx[exp_dist.cumsum() < epsilon][-1]
        else:
            left = 0
        right = range_idx[exp_dist.cumsum() > (exp_dist.sum() - epsilon)][0]
        return left, right

    for i in range(max_iters):
        #logger.info(f"\niteration: {iter_count}")
        count += 1
        # get Median of Distribution
        median = get_pivot(range_dist)
        # comparision function
        maxPseudoRev, maxSet, queryTimeLog = get_nn_set(v, p, median, prod, C, db = meta['db_BZ'], normConst = meta['normConst'], algo = 'general_case_BZ', feasibles = meta['feasibles'], queryTimeLog = 0)
        # Compare Set Revenue with bestSet provided, and replace bestSet if more optimal
        current_set_revenue = calcRev(maxSet, p, v, prod)
        if current_set_revenue > best_set_revenue:
            best_set, best_set_revenue = maxSet, current_set_revenue
        if (maxPseudoRev / v[0]) >= median:
            range_dist[range_idx >= median] += np.log(compstep_prob)
            range_dist[range_idx < median] += np.log(1 - compstep_prob)
        else:
            range_dist[range_idx <= median] += np.log(compstep_prob)
            range_dist[range_idx > median] += np.log(1 - compstep_prob)

        # shift all density from lower than best revenue got into upper end
        shift_density_total = np.sum(np.exp(range_dist[range_idx < best_set_revenue]))
        if (shift_density_total > 0):
            range_dist[range_idx < best_set_revenue] = np.log(0)
            range_dist[range_idx >= best_set_revenue] += np.log(
                    shift_density_total / len(range_dist[range_idx >= best_set_revenue]))
        # avoid overflows
        range_dist -= np.max(range_dist) 
        belief_start, belief_end = get_belief_interval(range_dist)
 
        if (belief_end - belief_start) <= early_termination_width:
            break

    timeTaken = time.time()-st

    
    print "\t\tAssortBZ-Z Opt Set Size:",len(best_set)
    print "\t\tAssortBZ-Z Opt Set:",best_set

    return best_set_revenue, best_set, timeTaken