import numpy as np
import time, pprint, pickle, datetime, random, os, itertools
from all_algos_test import get_real_price, generate_instance, get_log_dict,log_wrapper,overlap_wrapper

# Importing Algorithms

from capAst_oracle import capAst_oracle_general

from capAst_scikitLSH import capAst_AssortLSH_general, capAst_AssortExact_general, preprocess_general_assortx


def generate_instance_general(price_range,prod,genMethod,iterNum,C=None):

  p,v = generate_instance(price_range,prod,genMethod,iterNum)

  # # nchoosek
  # if prod <= 200 and C is not None:
  #   feasibles = list(itertools.combinations(range(n), k)) #TBD

  #arbitrary
  nsets = min(int(prod**1.5),int(1e7))
  feasibles = []
  for i in range(nsets):
    temp = random.randint(1,2**prod)
    temp2 = [int(x) for x in format(temp,'0'+str(prod)+'b')]
    feasibles.append(np.asarray(temp2))

  return p,v,feasibles


def main():

  #parameters required
  np.random.seed(10)
  price_range = 1000      #denotes highest possible price of a product
  eps         = 0.1       #tolerance
  N           =  30 #  #number of times Monte Carlo simulation will run
  prodList    = [100, 200] # [100,200, 400, 600, 800, 1000,5000,10000,20000] #[10000,20000] #[100,200] # 
  genMethod   = 'synthetic' #'bppData' #
  algos = {'Linear-Search':capAst_oracle_general}#{'Assort-Exact':capAst_AssortExact_general,'Assort-LSH':capAst_AssortLSH_general,'Linear-Search':capAst_oracle_general}
  benchmark = 'Linear-Search'


  loggs = get_log_dict(prodList,N,algos,price_range,eps,genMethod)
  loggs['additional']['lenfeasibles'] = np.zeros(N)
  badError = 0
  t1= time.time()
  for i,prod in enumerate(prodList):
      
    t0 = time.time()
    t = 0
    while(t<N):
        #generating the price 
        p,v,feasibles = generate_instance_general(price_range,prod,genMethod,t)
        # loggs['additional']['lenfeasibles'][t] = len(feasibles)

        meta = preprocess_general_assortx(prod,feasibles,p,eps)

        #run algos
        maxSet = None
        maxSetBenchmark = None
        for algoname in algos:
          print 'Executing ',algoname
          loggs[algoname]['rev'][i,t],maxSet,loggs[algoname]['time'][i,t] = algos[algoname](prod,feasibles,p,v,meta)
          if algoname==benchmark:
            maxSetBenchmark = maxSet

        loggs,badError = overlap_wrapper(benchmark,algos,loggs,i,t,badError,maxSet,maxSetBenchmark,eps)

        t = t+1    
        print 'Iteration number is ', str(t),' for prod size ',prod
      
    loggs = log_wrapper(algos,loggs,benchmark,i)

    
    print 'Calculations done for number of products'  
    print 'Time taken to run is', time.time() - t0,'\n\n\n'    

    #dump it incrementally for each product size
    # pickle.dump(loggs,open('./output/loggs_'+genMethod+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

  print 'Total experiment time taken is', time.time()  - t1
  for algoname in algos:
    print algoname
    print loggs[algoname]['time_mean']
    print loggs[algoname]['revPctErr_mean']


if __name__=='__main__':
  main()