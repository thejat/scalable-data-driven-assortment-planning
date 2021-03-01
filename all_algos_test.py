

import numpy as np
import time, pickle, datetime, random, os, copy, collections
from real_data import get_feasibles_realdata, get_feasibles_realdata_by_assortment
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import chain
from collections import Counter
import random

from competing_algos import capAst_static_mnl, capAst_LP, capAst_adxopt, genAst_oracle
from proposed_algos import capAst_AssortExact, capAst_AssortLSH, genAst_AssortLSH, genAst_AssortExact, preprocess, capAst_AssortBZ, genAst_AssortBZ

def get_real_prices(price_range, prod, iterNum = 0):
  fname = os.getcwd() + '/billion_price_data/processed_data/usa_2/numProducts_stats.npz'
  dateList = np.load(fname)['good_dates']
  fileName = os.getcwd() + '/billion_price_data/processed_data/usa_2/prices_'
  fileNameList = []
  for chosenDay in dateList:
    fileNameList.append(fileName+ chosenDay+'.npz')

  allPrices = np.load(fileNameList[iterNum])['arr_0']
  allPrices = allPrices[np.isfinite(allPrices)]
  # print allPrices
  allValidPrices = allPrices[allPrices > 0.01]
  allValidPrices = allValidPrices[allValidPrices < price_range]
  #allValidPrices = sorted(list(allValidPrices))
  p = allValidPrices[:prod]
  # p = random.sample(allValidPrices, prod)
  return p 

def generate_instance(price_range,prod,genMethod,iterNum):
  if genMethod=='bppData':
    p = get_real_prices(price_range, prod, iterNum)
  else:
    p = price_range * np.random.beta(1,1,prod)
  v = []  
  p = np.around(p, decimals =2)
  p = np.insert(p,0,0) #inserting 0 as the first element to denote the price of the no purchase option
  for i in p[1:]: 
     v.append(np.exp(-(4.61+0.00461*(i)+np.random.normal(0,0.1))))  
  v_0 = sum(v)*3.0/7
  v.insert(0,v_0)
#   #generating the customer preference vector, we don't care that it is in 0,1. Want it away from 0 for numeric. stability.
#   v = np.around(np.random.beta(1,5,prod+1) + 1e-3, decimals =7) #v is a prod+1 length vector as the first element signifies the customer preference for the no purchase option
#   #Ensure that there are no duplicate entires in v - required for Static-MNL.
#   u, indices = np.unique(v, return_inverse=True)   

#   while(not(len(u)== prod+1) or abs(v[0])<1e-3):
#       if abs(v[0])<1e-3:
#         v[0] = np.around(np.random.rand(1) + 1e-3,decimals =7)
#         u, indices = np.unique(v, return_inverse=True) 
#       extraSize = prod+1 - len(u)
#       newEnt = np.around(np.random.rand(extraSize)+1e-3,decimals=7)
#       v= np.concatenate((u,newEnt))
#       u, indices = np.unique(v, return_inverse=True)

  #print "instance max price:",max(p)
  return p,np.around(v,decimals=7)

def get_log_dict(prodList,N,algos,price_range,eps,C=None):

  def matrices(prodList,N):
    names1 = ['revPctErr','setOlp','corrSet','rev','time']
    names2 = ['corrSet_mean', 'setOlp_mean',  'revPctErr_max', 'revPctErr_mean','revPctErr_std', 'time_mean', 'time_std'] 
    output = {}
    for name in names1:
     output[name] = np.zeros((len(prodList), N))
    for name in names2: 
      output[name] = np.zeros(len(prodList)) 
    return output

  loggs = collections.OrderedDict()
  loggs['additional'] = {'prodList':prodList,'algonames':algos.keys(),'N':N,'eps':eps,'price_range':price_range}
  if C is not None:
    loggs['additional']['C'] = C
  else:
    loggs['additional']['C'] = np.zeros((len(prodList), N))

  for algoname in algos:
    loggs[algoname] = matrices(prodList,N)

    loggs[algoname]['maxSet'] = {}

  return loggs

def compute_summary_stats(algos,loggs,benchmark,i):
  for algoname in algos:
    # print algoname
    if benchmark in algos:
      loggs[algoname]['revPctErr'][i] = (loggs[benchmark]['rev'][i,:] - loggs[algoname]['rev'][i,:])/(loggs[benchmark]['rev'][i,:]+1e-6)
      loggs[algoname]['revPctErr_mean'][i] = np.mean(loggs[algoname]['revPctErr'][i,:])
      loggs[algoname]['revPctErr_std'][i] = np.std(loggs[algoname]['revPctErr'][i,:])
      loggs[algoname]['revPctErr_max'][i] = np.max(loggs[algoname]['revPctErr'][i,:])
    loggs[algoname]['corrSet_mean'][i] = np.mean(loggs[algoname]['corrSet'][i,:])
    loggs[algoname]['setOlp_mean'][i] = np.mean(loggs[algoname]['setOlp'][i,:])
    loggs[algoname]['time_mean'][i] = np.mean(loggs[algoname]['time'][i,:])
    loggs[algoname]['time_std'][i] = np.std(loggs[algoname]['time'][i,:])

  return loggs

def compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSetBenchmark,eps):

  def overlap(maxSet,maxSetBenchmark):
    setOlp  = len(maxSetBenchmark.intersection(maxSet))
    corrSet = int(setOlp==  len(maxSetBenchmark))
    setOlp  = setOlp*1.0/len(maxSetBenchmark) #to normalize
    return setOlp,corrSet

  if benchmark in algos:
    for algoname in algos:
      # print 'Collecting benchmarks for ',algoname
      loggs[algoname]['setOlp'][i,t],loggs[algoname]['corrSet'][i,t] = overlap(loggs[algoname]['maxSet'][(i,t)],maxSetBenchmark)
      if(loggs[benchmark]['rev'][i,t] - loggs[algoname]['rev'][i,t] > eps ):
          badError = badError +1
  return loggs,badError


def get_real_prices_parameters_by_product(subset):
    fname = os.getcwd() + '/freq_itemset_data/original_data/ta_feng_all_months_merged.csv'
    df = pd.read_csv(fname)
    le = LabelEncoder()
    df['ENCODED_PRODUCT_ID'] = le.fit_transform(df['PRODUCT_ID'])
    df = df[df['SALES_PRICE'] < 3200]
    dict1 = df.set_index('ENCODED_PRODUCT_ID').to_dict()['SALES_PRICE']
    df1 = df.groupby(['TRANSACTION_DT','CUSTOMER_ID'], as_index=False)['ENCODED_PRODUCT_ID'].agg(lambda x: list(x))
    df1['ENCODED_PRODUCT_ID'] = df1['ENCODED_PRODUCT_ID'].apply(lambda x:[ int(y) for y in x])  
    Y = pd.Series(Counter(chain.from_iterable(df1.ENCODED_PRODUCT_ID)))
    df2 = pd.DataFrame(Y)
    df2.reset_index(level=0, inplace=True)
    df2.columns = ['Product','Frequency']
    df2['v'] = df2['Frequency']/Y.sum()
    df2['price'] = df2['Product'].map(dict1) 
    dict2 = dict(zip(df2['Product'], df2[['v','price']].values.tolist()))
    v = []
    p = []
    for i in subset:
        v.append(dict2[i][0])
        p.append(dict2[i][1])
    #+np.random.normal(0,10**-7)    
    #v_0 = df2['v'].sum()*(3.0/7)
    v_0 = sum(v)*(3.0/7)
    v.insert(0,v_0)
    p.insert(0,0)
    
    return np.array(p), np.array(v)
    
def generate_instance_general(price_range,prod,genMethod,iterNum,lenFeas=None,real_data=None):
    
    
    if genMethod == 'tafeng' and lenFeas != None :
        feasibles, C, subset = get_feasibles_realdata_by_assortment(os.getcwd() +'/freq_itemset_data/tafeng_final_0p00001_txns119390.txt',lenFeas, isCSV=False,min_ast_length=8)
    
    if real_data is None and genMethod != "tafeng" :
        if lenFeas is None:
            nsets = int(prod**1.5)
        else:
            nsets = lenFeas
    #synthetic
        feasibles = []
        C = 0
        for i in range(nsets):
            temp = random.randint(1,2**prod-1)
            temp2 = [int(x) for x in format(temp,'0'+str(prod)+'b')]
            set_char_vector = np.asarray(temp2)
            feasibles.append(set_char_vector)
            C = max(C,np.sum(set_char_vector))
    elif real_data is not None:
    #real
        feasibles,C,subset = get_feasibles_realdata(fname=real_data['fname'],isCSV=real_data['isCSV'],min_ast_length=real_data['min_ast_length'])
        prod = len(subset)
    
    if genMethod == 'tafeng':
      p,v = get_real_prices_parameters_by_product(subset)
      return p,v,feasibles,int(C),len(subset)
    else: 
      p,v = generate_instance(price_range,prod,genMethod,iterNum)
      #if real_data is not None:  
      #    v[0] = sum(v[1:])*(9.0)   
      return p,v,feasibles,int(C),prod 
         
    

def run_prod_experiment(flag_capacitated=True,flag_savedata=True,genMethod='synthetic'):

  #parameters required
  random.seed(10)
  np.random.seed(1000)
  price_range = 1000      #denotes highest possible price of a product
  eps         = 0.1       #tolerance
  N           = 50 #   #number of times Monte Carlo simulation will run
  if flag_capacitated == True:
    C           = 50      #capacity of assortment in [10,20,50,100,200]
    if genMethod=='synthetic':
      prodList    = [15000,20000] #[100,200,300] #
    else:
      prodList    = [100, 250, 500, 1000, 3000, 5000, 7000,10000,15000]
        
    #algos = collections.OrderedDict({'LP':capAst_LP})
    algos = collections.OrderedDict({'Adxopt':capAst_adxopt})#,'LP':capAst_LP,'Assort-Exact':capAst_AssortExact'Adxopt':capAst_adxopt,'Assort-Exact':capAst_AssortExact, 'Static-MNL':capAst_paat}
    benchmark = 'LP'#'Static-MNL'#
    loggs = get_log_dict(prodList,N,algos,price_range,eps,C)

  else:
    prodList    = [100,200,400,800,1600]  
    algos       = collections.OrderedDict({'Linear-Search':genAst_oracle,'Assort-Exact-G':genAst_AssortExact,'Assort-LSH-G':genAst_AssortLSH})
    benchmark   = 'Linear-Search'
    loggs = get_log_dict(prodList,N,algos,price_range,eps)
    loggs['additional']['lenFeasibles'] = np.zeros(len(prodList))


  badError = 0
  t1= time.time()
  for i,prod in enumerate(prodList):
      
    t0 = time.time()
    t = 0
    while(t<N):

      print 'Iteration number is ', str(t+1),' of ',N,', for prod size ',prod

      #generating the price
      meta = {'eps':eps}
      if flag_capacitated == True:
        #file_1 = open("products.pkl",'rb')   
        #product_choices = pickle.load(file_1)
        #choices = random.sample(product_choices,prod)
        #subset = choices  
        #p,v = get_real_prices_parameters_by_product(subset)
        #C = prod
        p,v = generate_instance(price_range,prod,genMethod,t)
      else:
        p,v,feasibles,C,prod = generate_instance_general(price_range,prod,genMethod,t)
        loggs['additional']['C'][i,t] = C
        meta['feasibles'] = feasibles

      #preprocessing for proposed algos
      if 'Assort-Exact' in algos:
        meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'special_case_exact')
      if 'Assort-LSH' in algos:
        meta['db_LSH'],_,_ = preprocess(prod, C, p, 'special_case_LSH', nEst=20,nCand=80)#Hardcoded values
      if 'Assort-BZ' in algos:
        meta['db_BZ'],_,meta['normConst'] = preprocess(prod, C, p, 'special_case_BZ', nEst=20,nCand=80)#Hardcoded values  
      if 'Assort-Exact-G' in algos:
        meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_exact',feasibles=feasibles)
      if 'Assort-LSH-G' in algos:
        meta['db_LSH'],_,_ = preprocess(prod, C, p, 'general_case_LSH', nEst=20,nCand=80,feasibles=feasibles)#Hardcoded values
      if 'Assort-BZ-G' in algos:
        meta['db_BZ'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_BZ', nEst=20,nCand=80,feasibles=feasibles)#Hardcoded values


      #run algos
      maxSetBenchmark = None
      for algoname in algos:
        print '\tExecuting ',algoname
        loggs[algoname]['rev'][i,t],loggs[algoname]['maxSet'][(i,t)],loggs[algoname]['time'][i,t] = algos[algoname](prod,C,p,v,meta)
        print '\t\tTime taken is ',loggs[algoname]['time'][i,t],'sec.'

        if algoname==benchmark:
          maxSetBenchmark = copy.deepcopy(loggs[algoname]['maxSet'][(i,t)])

      loggs,badError = compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSetBenchmark,eps)

      t = t+1    
      

    
    print 'Experiments (',N,' sims) for number of products ',prod, ' is done.'  
    print 'Cumulative time taken is', time.time() - t0,'\n'   
    loggs = compute_summary_stats(algos,loggs,benchmark,i)
    if flag_capacitated != True:
      loggs['additional']['lenFeasibles'][i] = len(feasibles)

    #dump it incrementally for each product size
    if flag_savedata == True:
      if flag_capacitated == True:
        pickle.dump(loggs,open('./output/cap_loggs_'+genMethod+'_prod_'+str(prod)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))
      else:
        pickle.dump(loggs,open('./output/gen_loggs_'+genMethod+'_prod_'+str(prod)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

  print '\nAll experiments done. Total time taken is', time.time()  - t1,'\n\n'
  print "Summary:"
  for algoname in algos:
    print '\t',algoname,'time_mean',loggs[algoname]['time_mean']
    print '\t',algoname,'revPctErr_mean',loggs[algoname]['revPctErr_mean']

  return loggs


def run_lenFeas_experiment(flag_savedata=True,genMethod='synthetic',nEst=20,nCand=80):

  #parameters required
  random.seed(10)
  np.random.seed(1000)
  price_range = 1000      #denotes highest possible price of a product
  eps         = 1       #tolerance
  N           = 50 #   #number of times Monte Carlo simulation will run
  prod        = 1000
  lenFeasibles= [100,200,400,800,1600,3200,6400,12800,25600,51200]
  #lenFeasibles= [51200]
  algos       = collections.OrderedDict({'Linear-Search':genAst_oracle, 'Assort-LSH-G':genAst_AssortLSH, 'Assort-Exact-G':genAst_AssortExact, 'Assort-BZ-G' : genAst_AssortBZ})
  #algos       = collections.OrderedDict({'Linear-Search':genAst_oracle,'Assort-LSH-G':genAst_AssortLSH,'Assort-Exact-G':genAst_AssortExact, 'Assort-BZ-G' : genAst_AssortBZ})
  benchmark   = 'Linear-Search'
  loggs = get_log_dict(lenFeasibles,N,algos,price_range,eps) #hack
  loggs['additional']['lenFeasibles'] = lenFeasibles
  loggs['additional']['nEst'] = nEst
  loggs['additional']['nCand'] = nCand


  badError = 0
  t1= time.time()
  for i,lenFeas in enumerate(lenFeasibles):
      
    t0 = time.time()
    t = 0
    while(t<N):

      print 'Iteration number is ', str(t+1),' of ',N,', for no. of assortments ',lenFeas

      #generating the price
      meta = {'eps':eps}
      p,v,feasibles,C,prod = generate_instance_general(price_range,prod,genMethod,t,lenFeas=lenFeas)
      loggs['additional']['C'][i,t] = C
      meta['feasibles'] = feasibles

      #preprocessing for proposed algos
      if 'Assort-Exact-G' in algos:
        meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_exact',feasibles=feasibles)
      if 'Assort-LSH-G' in algos:
        meta['db_LSH'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_LSH', nEst=nEst,nCand=nCand,feasibles=feasibles)#Hardcoded values
      if 'Assort-BZ-G' in algos:
        meta['db_BZ'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_BZ', nEst=nEst,nCand=nCand,feasibles=feasibles)#Hardcoded values  
        


      #run algos
      maxSetBenchmark = None
      for algoname in algos:
        print '\tExecuting ',algoname
        loggs[algoname]['rev'][i,t],loggs[algoname]['maxSet'][(i,t)],loggs[algoname]['time'][i,t] = algos[algoname](prod,C,p,v,meta)
        print '\t\tTime taken is ',loggs[algoname]['time'][i,t],'sec.'

        if algoname==benchmark:
          maxSetBenchmark = copy.deepcopy(loggs[algoname]['maxSet'][(i,t)])

      loggs,badError = compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSetBenchmark,eps)

      t = t+1    
      

    
    print 'Experiments (',N,' sims) for number of feasibles ',lenFeas, ' is done.'  
    print 'Cumulative time taken is', time.time() - t0,'\n'   
    loggs = compute_summary_stats(algos,loggs,benchmark,i)

    #dump it incrementally for each product size
    if flag_savedata == True:
      pickle.dump(loggs,open('./output/gen_loggs_'+genMethod+'_lenF_'+str(lenFeas)+'_nCand_'+str(nCand)+'_nEst_'+str(nEst)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

  print '\nAll experiments done. Total time taken is', time.time()  - t1,'\n\n'
  print "Summary:"
  for algoname in algos:
    print '\t',algoname,'time_mean',loggs[algoname]['time_mean']
    print '\t',algoname,'revPctErr_mean',loggs[algoname]['revPctErr_mean']

  return loggs

def run_real_ast_experiment(flag_savedata=True,nEst=20,nCand=80):

  #parameters required
  np.random.seed(1000)
  price_range = 1000      #denotes highest possible price of a product
  eps         = 1       #tolerance
  N           = 50 #   #number of times Monte Carlo simulation will run
  real_data_list = [
    {'fname':'freq_itemset_data/retail0p0001_240852_txns88162.csv','isCSV':True,'min_ast_length':3},
    {'fname':'freq_itemset_data/foodmartFIM0p0001_233231_txns4141.csv','isCSV':True,'min_ast_length':4},
    {'fname':'freq_itemset_data/chains0p00001_txns1112949.txt','isCSV':False,'min_ast_length':5},
    {'fname':'freq_itemset_data/OnlineRetail0p000001_txns540455.txt','isCSV':False,'min_ast_length':3}]
  #real_data_list = [{'fname':'freq_itemset_data/tafeng_final_0p00001_txns119390.txt','isCSV':False,'min_ast_length':8}]
  algos       = collections.OrderedDict({'Linear-Search':genAst_oracle,'Assort-LSH-G':genAst_AssortLSH,'Assort-Exact-G':genAst_AssortExact, 'Assort-BZ-G':genAst_AssortBZ})
  benchmark   = 'Linear-Search'
  loggs = get_log_dict(real_data_list,N,algos,price_range,eps) #hack
  loggs['additional']['real_data_list'] = real_data_list



  badError = 0
  t1= time.time()
  for i,real_data in enumerate(real_data_list):
      
    t0 = time.time()
    t = 0
    while(t<N):

      print 'Iteration number is ', str(t+1),' of ',N,', for real ast data ',real_data['fname']

      #generating the price
      meta = {'eps':eps}
    
    
      if genMethod =='tafeng':
          p,v,feasibles,C,prod = generate_instance_general(price_range,None,'tafeng',t,lenFeas=None,real_data=real_data)
      else:      
          p,v,feasibles,C,prod = generate_instance_general(price_range,None,'synthetic',t,lenFeas=None,real_data=real_data)  
      #
      loggs['additional']['C'][i,t] = C
      meta['feasibles'] = feasibles
      

      #preprocessing for proposed algos
      if 'Assort-Exact-G' in algos:
        meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_exact',feasibles=feasibles)
      if 'Assort-LSH-G' in algos:
        meta['db_LSH'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_LSH', nEst=nEst,nCand=nCand,feasibles=feasibles)#Hardcoded values
      if 'Assort-BZ-G' in algos:
        meta['db_BZ'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_BZ', nEst=nEst,nCand=nCand,feasibles=feasibles)#Hardcoded values  

      #run algos
      maxSetBenchmark = None
      for algoname in algos:
        print '\tExecuting ',algoname
        loggs[algoname]['rev'][i,t],loggs[algoname]['maxSet'][(i,t)],loggs[algoname]['time'][i,t] = algos[algoname](prod,C,p,v,meta)
        print '\t\tTime taken is ',loggs[algoname]['time'][i,t],'sec.'

        if algoname==benchmark:
          maxSetBenchmark = copy.deepcopy(loggs[algoname]['maxSet'][(i,t)])

      loggs,badError = compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSetBenchmark,eps)

      t = t+1    

    
    print 'Experiments (',N,' sims) for real ast data ',real_data['fname'], ' is done.'  
    print 'Cumulative time taken is', time.time() - t0,'\n'   
    loggs = compute_summary_stats(algos,loggs,benchmark,i)

    #dump it incrementally for each product size
    if flag_savedata == True:
      pickle.dump(loggs,open('./output/gen_loggs_real_ast_upto'+str(i)+'_nCand_'+str(nCand)+'_nEst_'+str(nEst)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

  print '\nAll experiments done. Total time taken is', time.time()  - t1,'\n\n'
  print "Summary:"
  for algoname in algos:
    print '\t',algoname,'time_mean',loggs[algoname]['time_mean']
    print '\t',algoname,'revPctErr_mean',loggs[algoname]['revPctErr_mean']

  return loggs


def run_prod_experiment_static_mnl(flag_capacitated=True,flag_savedata=True,genMethod='synthetic'):

  #parameters required
  random.seed(10)
  np.random.seed(1000)
  price_range = 1000      #denotes highest possible price of a product
  eps         = 0.1       #tolerance
  N           = 50 #   #number of times Monte Carlo simulation will run
  if flag_capacitated == True:
    C           = 100        #capacity of assortment
    if genMethod=='synthetic':
      prodList    = [100, 250, 500, 1000] #[100,200,300] #
    else:
      prodList    = [100, 250, 500, 1000]
    algos = collections.OrderedDict({'Static-MNL':capAst_static_mnl, 'LP':capAst_LP}) #'LP':capAst_LP
    benchmark = 'LP'
    loggs = get_log_dict(prodList,N,algos,price_range,eps,C)

  else:
    prodList    = [100,200,400,800,1600]
    algos       = collections.OrderedDict({'Linear-Search':genAst_oracle,'Assort-Exact-G':genAst_AssortExact,'Assort-LSH-G':genAst_AssortLSH})
    benchmark   = 'Linear-Search'
    loggs = get_log_dict(prodList,N,algos,price_range,eps)
    loggs['additional']['lenFeasibles'] = np.zeros(len(prodList))


  badError = 0
  t1= time.time()
  for i,prod in enumerate(prodList):
      
    t0 = time.time()
    t = 0
    while(t<N):

      print 'Iteration number is ', str(t+1),' of ',N,', for prod size ',prod

      #generating the price
      meta = {'eps':eps}
      if flag_capacitated == True:
          if genMethod = 'tafeng':  
              file_1 = open("products_final.pkl",'rb')
              product_choices = pickle.load(file_1)
              choices = random.sample(product_choices,prod)  
              p,v = get_real_prices_parameters_by_product(choices)
          else:  
              p,v = generate_instance(price_range,prod,genMethod,t)
      else:
        p,v,feasibles,C,prod = generate_instance_general(price_range,prod,genMethod,t)
        loggs['additional']['C'][i,t] = C
        meta['feasibles'] = feasibles

      #preprocessing for proposed algos
      if 'Assort-Exact' in algos:
        meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'special_case_exact')
      if 'Assort-LSH' in algos:
        meta['db_LSH'],_,_ = preprocess(prod, C, p, 'special_case_LSH', nEst=20,nCand=80)#Hardcoded values
      if 'Assort-Exact-G' in algos:
        meta['db_exact'],_,meta['normConst'] = preprocess(prod, C, p, 'general_case_exact',feasibles=feasibles)
      if 'Assort-LSH-G' in algos:
        meta['db_LSH'],_,_ = preprocess(prod, C, p, 'general_case_LSH', nEst=20,nCand=80,feasibles=feasibles)#Hardcoded values



      #run algos
      maxSetBenchmark = None
      for algoname in algos:
        print '\tExecuting ',algoname
        loggs[algoname]['rev'][i,t],loggs[algoname]['maxSet'][(i,t)],loggs[algoname]['time'][i,t] = algos[algoname](prod,C,p,v,meta)
        print '\t\tTime taken is ',loggs[algoname]['time'][i,t],'sec.'

        if algoname==benchmark:
          maxSetBenchmark = copy.deepcopy(loggs[algoname]['maxSet'][(i,t)])

      loggs,badError = compute_overlap_stats(benchmark,algos,loggs,i,t,badError,maxSetBenchmark,eps)

      t = t+1    
      
    print 'Experiments (',N,' sims) for number of products ',prod, ' is done.'  
    print 'Cumulative time taken is', time.time() - t0,'\n'   
    loggs = compute_summary_stats(algos,loggs,benchmark,i)
    if flag_capacitated != True:
      loggs['additional']['lenFeasibles'][i] = len(feasibles)

    #dump it incrementally for each product size
    if flag_savedata == True:
      if flag_capacitated == True:
        pickle.dump(loggs,open('./output/cap_loggs_'+genMethod+'_prod_'+str(prod)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))
      else:
        pickle.dump(loggs,open('./output/gen_loggs_'+genMethod+'_prod_'+str(prod)+'_'+datetime.datetime.now().strftime("%Y%m%d_%I%M%p")+'.pkl','wb'))

  print '\nAll experiments done. Total time taken is', time.time()  - t1,'\n\n'
  print "Summary:"
  for algoname in algos:
    print '\t',algoname,'time_mean',loggs[algoname]['time_mean']
    print '\t',algoname,'revPctErr_mean',loggs[algoname]['revPctErr_mean']

  return loggs





if __name__=='__main__':


  #1. General case, dependense on lsh parameters: bpp data and synthetic data

  # loggs5 = run_lenFeas_experiment(flag_savedata = True,genMethod='synthetic',nEst=40,nCand=160)
  # loggs5 = run_lenFeas_experiment(flag_savedata = True,genMethod='synthetic',nEst=100,nCand=200)
  # loggs5 = run_lenFeas_experiment(flag_savedata = True,genMethod='synthetic',nEst=20,nCand=80)
  # loggs6 = run_lenFeas_experiment(flag_savedata = True,genMethod='bppData',nEst=40,nCand=160)
  # loggs6 = run_lenFeas_experiment(flag_savedata = True,genMethod='bppData',nEst=100,nCand=200)
  # loggs6 = run_lenFeas_experiment(flag_savedata = True,genMethod='bppData',nEst=20,nCand=80)
   loggs6 = run_lenFeas_experiment(flag_savedata = True,genMethod='bppData',nEst=20,nCand=80)


  #2. General case: frequent itemset data
  
  #loggs7 = run_real_ast_experiment(flag_savedata = True, nEst=20,nCand=80)

  #3. Special case (cap constrained): bpp data and synthetic data
  #  loggs2 = run_prod_experiment(flag_capacitated = True,flag_savedata = True,genMethod='tafeng')

  # loggs1 = run_prod_experiment(flag_capacitated = True,flag_savedata = True,genMethod='synthetic')
  #  loggs2 = run_prod_experiment(flag_capacitated = True,flag_savedata = True,genMethod='bppData')
  ## loggs3 = run_prod_experiment(flag_capacitated = False,flag_savedata = True,genMethod='synthetic')
  ## loggs4 = run_prod_experiment(flag_capacitated = False,flag_savedata = True,genMethod='bppData')


  # loggs2 = run_prod_experiment_static_mnl(flag_capacitated = True,flag_savedata = True,genMethod='tafeng')
  