import numpy as np
import time, pickle, datetime,random
from all_algos_test import get_real_prices, generate_instance, get_log_dict, compute_summary_stats, compute_overlap_stats
from plots_paper import get_plots

# Importing Algorithms
from capAst_oracle import genAst_oracle
from capAst_scikitLSH import genAst_AssortLSH, genAst_AssortExact, preprocess


def generate_instance_general(price_range,prod,genMethod,iterNum,C=None):

  p,v = generate_instance(price_range,prod,genMethod,iterNum)


  #arbitrary
  nsets = min(int(prod**1.5),int(1e7))
  feasibles = []
  C = 0
  for i in range(nsets):
    temp = random.randint(1,2**prod-1)
    temp2 = [int(x) for x in format(temp,'0'+str(prod)+'b')]
    set_char_vector = np.asarray(temp2)
    feasibles.append(set_char_vector)
    C = max(0,np.sum(set_char_vector))

  return p,v,feasibles,int(C)


def main():

  #parameters required
  flag_savedata = True
  random.seed(10)
  np.random.seed(10)
  price_range = 1000        #denotes highest possible price of a product
  eps         =  0.1        #tolerance
  N           =  30 #2#     #number of times Monte Carlo simulation will run
  prodList    = [100,200,400,600,800,1000,5000,10000] #[100,200] # [100,200,400,600,800, 1000] #
  genMethod   = 'synthetic' #'bppData'#
  algos       = {'Linear-Search':genAst_oracle,'Assort-Exact-G':genAst_AssortExact,'Assort-LSH-G':genAst_AssortLSH}
  benchmark   = 'Linear-Search'


  loggs = get_log_dict(prodList,N,algos,price_range,eps,genMethod)
  loggs['additional']['lenfeasibles'] = np.zeros(len(prodList))
  badError = 0
  t1= time.time()
  for i,prod in enumerate(prodList):
      
    t0 = time.time()
    t = 0
    while(t<N):

        print 'Iteration number is ', str(t+1),' of ',N,', for prod size ',prod


        #generating the price 
        p,v,feasibles,C = generate_instance_general(price_range,prod,genMethod,t)
        # print feasibles[:1],len(feasibles[0]),C
        loggs['additional']['C'][i,t] = C

        meta = {'eps':eps,'feasibles':feasibles}

        if 'Assort-Exact-G' in algos:
          meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_exact',feasibles=feasibles)
        if 'Assort-LSH-G' in algos:
          meta['db_LSH'],_,_ = preprocess(prod, C, p, 'general_case_LSH', nEst=100,nCand=200,feasibles=feasibles)#Hardcoded values

        #run algos
        maxSet,maxSetBenchmark = None, None
        for algoname in algos:
          
          loggs[algoname]['rev'][i,t],maxSet,loggs[algoname]['time'][i,t] = algos[algoname](prod,C,p,v,meta)
          print '\tExecuted ',algoname,' in ',loggs[algoname]['time'][i,t],'sec.'#,' Set:',maxSet
          if algoname==benchmark:
            maxSetBenchmark = maxSet
        loggs,badError = compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSet,maxSetBenchmark,eps)

        t = t+1    
      

    
    print 'Experiments (',N,' sims) for number of products ',prod, ' is done.'  
    print 'Cumulative time taken is', time.time() - t0,'\n'    
    loggs = compute_summary_stats(algos,loggs,benchmark,i)
    loggs['additional']['lenfeasibles'][i] = len(feasibles)

    #dump it incrementally for each product size
    if flag_savedata == True:
      pickle.dump(loggs,open('./output/gen_loggs_'+genMethod+'_'+str(prod)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

  print '\nAll experiments done. Total time taken is', time.time()  - t1,'\n\n'
  print "Summary:"
  for algoname in algos:
    print '\t',algoname,'time_mean',loggs[algoname]['time_mean']
    print '\t',algoname,'revPctErr_mean',loggs[algoname]['revPctErr_mean']

  return loggs

if __name__=='__main__':
  loggs = main()
  # get_plots(fname=None,flag_savefig=False,xlim=5001,loggs=loggs)