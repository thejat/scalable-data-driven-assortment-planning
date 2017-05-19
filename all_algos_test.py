import numpy as np
import time, pprint, pickle, datetime, random, os

# Importing Algorithms

from capAst_paat import capAst_paat
def capAst_paat2(prod,C,p,v,meta=None):
  if prod > 1000:
    print '\n\n\nPAAT will NOT execute for number of products greater than 1000******\n'
    print 'using capAst_LP instead\n\n'
    rev, maxSet, time = capAst_LP(prod, C, p, v)
  else:
    rev, maxSet, time = capAst_paat(prod, C, p, v)
    maxSet = set(maxSet.astype(int))
  return rev,maxSet,time

from capAst_LP import capAst_LP
def capAst_LP2(prod,C,p,v,meta=None):
  rev, maxSet, time = capAst_LP(prod, C, p, v)
  return rev,maxSet,time

from capAst_oracle import capAst_oracle
def capAst_oracle2(prod,C,p,v,meta=None):
  rev, maxSet, time = capAst_oracle(prod, C, p, v)
  return rev,maxSet,time

from capAst_adxopt import capAst_adxopt
def capAst_adxopt2(prod,C,p,v,meta=None):
  rev, maxSet, time = capAst_adxopt(prod, C, p, v)
  return rev,maxSet,time

from capAst_scikitLSH import capAst_NNalgo, preprocess
#Preprocessing for Assort-Exact and Assort-LSH
def preprocess_assortx(prod,C,p,eps):
  nEst = 100
  nCand = 200
  _, dbListskLSH,_,_ = preprocess(prod, C, p,  eps,  'skLSH_singleLSH', nEst=nEst,nCand=nCand)    
  KList, dbListExact, _, normConstList = preprocess(prod, C, p,  eps,  'exactNN_single')
  return {'dbListskLSH':dbListskLSH,'dbListExact':dbListExact,'KList':KList,'normConstList':normConstList,'eps':eps,'nEst':nEst,'nCand':nCand}
# Assort-Exact
def capAst_AssortExact(prod,C,p,v,meta):
  rev, maxSet, time = capAst_NNalgo(prod, C, p, v, 
    meta['eps'],  
    algo = 'exactNN_single',
    preprocessed =True,
    KList=meta['KList'], 
    dbList=meta['dbListExact'], 
    normConstList=meta['normConstList'])
  maxSet = set(maxSet.astype(int))
  return rev,maxSet,time
# Assort-LSH
def capAst_AssortLSH(prod,C,p,v,meta):
  rev, maxSet, time = capAst_NNalgo(prod, C, p, v,
    meta['eps'],  
    algo = 'skLSH_singleLSH', 
    nEst =meta['nEst'], 
    nCand =meta['nCand'] , 
    preprocessed =True,
    KList=meta['KList'], 
    dbList =meta['dbListskLSH'], 
    normConstList=meta['normConstList']) 
  return rev,maxSet,time

#parameters required
np.random.seed(10)
C           = 50 # 10 # #capacity of assortment
price_range = 1000      #denotes highest possible price of a product
eps         = 0.1       #tolerance
N           = 30 # 2 #  #number of times Monte Carlo simulation will run
prodList    = [100, 200, 400, 600, 800, 1000,5000,10000,20000] #[10000,20000] #[100,200] # 
genMethod   = 'synthetic' #'bppData' #


def get_real_price(price_range, prod, iterNum = 0):
  fname = os.getcwd() + '/billion_price_data/processed_data/usa_2/numProducts_stats.npz'
  dateList = np.load(fname)['good_dates']
  fileName = os.getcwd() + '/billion_price_data/processed_data/usa_2/prices_'
  fileNameList = []
  for chosenDay in dateList:
    fileNameList.append(fileName+ chosenDay+'.npz')

  allPrices = np.load(fileNameList[iterNum])['arr_0']
  allValidPrices = allPrices[allPrices < price_range]
  p = random.sample(allValidPrices, prod)
  return p 



algos = {'assort-exact':capAst_AssortExact,'assort-lsh':capAst_AssortLSH,'adxopt':capAst_adxopt2,'lp':capAst_LP2,'paat':capAst_paat2}
benchmark = 'paat'

def generate_instance(price_range,prod,genMethod,iterNum):
  if genMethod=='bppData':
    p = get_real_price(price_range, prod, iterNum)
  else:
    p = price_range * np.random.beta(2,5,prod) 
  p = np.around(p, decimals =2)
  p = np.insert(p,0,0) #inserting 0 as the first element to denote the price of the no purchase option
  
  #generating the customer preference vector
  v = np.random.rand(prod+1) #v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
  v = np.around(v, decimals =7)                
  #Ensure that there are no duplicate entires in v - required for Paat's algorithm
  u, indices = np.unique(v, return_inverse=True)   
  ct = 1
  while(not(len(u)== prod+1) or v[0]==0):
      if v[0]==0:
        v[0] = np.random.rand(1)
        u, indices = np.unique(v, return_inverse=True) 
      #print len(u)
      extraSize = prod+1 - len(u)
      newEnt = np.random.rand(extraSize)
      newEnt = np.around(newEnt, decimals =2) 
      v= np.concatenate((u,newEnt))
      u, indices = np.unique(v, return_inverse=True)
      ct =ct+1
  print 'Number of times v had to be generated', ct

  return p,v

def get_log_dict(prodList,N,algos,C,price_range,eps):

  def matrices(prodList,N):
    names1 = ['revPctErr','setOlp','corrSet','rev','time']
    names2 = ['corrSet_mean', 'setOlp_mean',  'revPctErr_max', 'revPctErr_mean','revPctErr_std', 'time_mean', 'time_std'] 
    output = {}
    for name in names1:
     output[name] = np.zeros((np.shape(prodList)[0], N))
    for name in names2: 
      output[name] = np.zeros( np.shape(prodList)[0]) 
    return output

  loggs = {}
  loggs['additional'] = {'prodList':prodList,'algonames':algos.keys(),'N':N,'eps':eps,'price_range':price_range,'C':C}
  for algoname in algos:
    loggs[algoname] = matrices(prodList,N)
  return loggs

def log_wrapper(algos,loggs,benchmark,i):
  for algoname in algos:
    print algoname
    if benchmark in algos:
      loggs[algoname]['revPctErr'][i] = (loggs[benchmark]['rev'][i,:] - loggs[algoname]['rev'][i,:])/loggs[benchmark]['rev'][i,:]
      loggs[algoname]['revPctErr_mean'][i] = np.mean(loggs[algoname]['revPctErr'][i,:])
      loggs[algoname]['revPctErr_std'][i] = np.std(loggs[algoname]['revPctErr'][i,:])
      loggs[algoname]['revPctErr_max'][i] = np.max(loggs[algoname]['revPctErr'][i,:])
    loggs[algoname]['corrSet_mean'][i] = np.mean(loggs[algoname]['corrSet'][i,:])
    loggs[algoname]['setOlp_mean'][i] = np.mean(loggs[algoname]['setOlp'][i,:])
    loggs[algoname]['time_mean'][i] = np.mean(loggs[algoname]['time'][i,:])
    loggs[algoname]['time_std'][i] = np.std(loggs[algoname]['time'][i,:])

  return loggs

def overlap_wrapper(benchmark,algos,loggs,i,t,badError,maxSet,maxSetBenchmark):
  if benchmark in algos:
    for algoname in algos:
      print algoname
      loggs[algoname]['setOlp'][i,t],loggs[algoname]['corrSet'][i,t] = overlap(maxSet,maxSetBenchmark)
      if(loggs[benchmark]['rev'][i,t] - loggs[algoname]['rev'][i,t] > eps ):
          badError = badError +1
  return loggs,badError

def overlap(maxSet,maxSetBenchmark):
  setOlp  = len(maxSetBenchmark.intersection(maxSet))
  corrSet = int(setOlp==  len(maxSetBenchmark))
  setOlp  = setOlp/len(maxSetBenchmark) #to normalize
  return setOlp,corrSet


def main():
  loggs = get_log_dict(prodList,N,algos,C,price_range,eps)
  badError = 0
  t1= time.time()
  for i,prod in enumerate(prodList):
      
    t0 = time.time()
    t = 0
    while(t<N):
        #generating the price 
        p,v = generate_instance(price_range,prod,genMethod,t)

        meta = preprocess_assortx(prod,C,p,eps) #tbd: make it conditional

        #run algos
        for algoname in algos:
          print algoname
          loggs[algoname]['rev'][i,t],maxSet,loggs[algoname]['time'][i,t] = algos[algoname](prod,C,p,v,meta)
          if algoname==benchmark:
            maxSetBenchmark = maxSet

        loggs,badError = overlap_wrapper(benchmark,algos,loggs,i,t,badError,maxSet,maxSetBenchmark)

        t = t+1    
        print 'Iteration number is ', str(t)
      
    loggs = log_wrapper(algos,loggs,benchmark,i)

    
    print 'Calculations done for number of products', prod  
    print 'Time taken to run is', time.time() - t0    

    #dump it incrementally for each product size
    pickle.dump(loggs,open('./output/loggs_'+genMethod+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

  print 'Total experiment time taken is', time.time()  - t1
  for algoname in algos:
    print algoname
    print loggs[algoname]['time_mean']
    print loggs[algoname]['revPctErr_mean']


if __name__=='__main__':
  main()