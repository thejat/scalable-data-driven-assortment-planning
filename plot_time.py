# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:33:32 2016

@author: Deeksha
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:40:55 2016

@author: Deeksha
"""
import numpy as np
import matplotlib.pyplot as plt

params = {'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

prodList = [100, 200, 400, 600, 800, 1000, 2000, 5000, 10000,15000,20000]
nCandList= [50,200]

price_range = [1000]
C = [50]
numComb = np.shape(C)[0]
fileNames= []

timeLSHNN_mean , timeLSHNN_std, timePaat_mean, timePaat_std, timeExactNN_mean, timeExactNN_std = ( np.zeros((numComb, np.shape(prodList)[0])) for i in range(6))

filePath= 'algoStats_theja20170319/stats_pmax_1000_C_50/'

for i in range(numComb):
    fileNames.append(filePath + 'allStats_pmax_' + str(price_range[i])+'_C_'+str(C[i]) + '.npz')

for i in range(numComb):
    timeLSHNN_mean[i,:] = np.load(fileNames[i])['timeLSHNN_mean'][:,1]
    timeLSHNN_std[i,:] = np.load(fileNames[i])['timeLSHNN_std'][:,1]
    timePaat_mean[i,:] = np.load(fileNames[i])['timePaat_mean']
    timePaat_std[i,:] = np.load(fileNames[i])['timePaat_std']
    timeExactNN_mean[i,:] = np.load(fileNames[i])['timeExactNN_mean']
    timeExactNN_std[i,:] = np.load(fileNames[i])['timeExactNN_std']

#revPctErrExactNN_mean, revPctErrExactNN_std, revPctErrExactNN_max, corrSetExactNN_mean, setOlpExactNN_mean = ( np.zeros((numComb, np.shape(prodList)[0])) for i in range(5))
#revPctErrLSHNN_mean, revPctErrLSHNN_std, revPctErrLSHNN_max, corrSetLSHNN_mean, setOlpLSHNN_mean = ( np.zeros((numComb, np.shape(prodList)[0], np.shape(nCandList)[0])) for i in range(5))

#for i in range(np.shape(prodList)[0]):
#     prod = prodList[i]
#     revPctErrExactNN_mean[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['revPctErrExactNN_mean']
#     revPctErrExactNN_std[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['revPctErrExactNN_std']
#     revPctErrExactNN_max[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['revPctErrExactNN_max']
#     corrSetExactNN_mean[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['corrSetExactNN_mean']
#     setOlpExactNN_mean[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['setOlpExactNN_mean']
#    
#     revPctErrLSHNN_mean[i,:] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['revPctErrLSHNN_mean']
#     revPctErrLSHNN_std[i,:] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['revPctErrLSHNN_std']
#     revPctErrLSHNN_max[i,:] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['revPctErrLSHNN_max']
#     corrSetLSHNN_mean[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['corrSetLSHNN_mean']
#     setOlpLSHNN_mean[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['setOlpLSHNN_mean']
##
#     timeExactNN_mean[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['timeExactNN_mean']
#     timeExactNN_std[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['timeExactNN_std']
#  
#     timeLSHNN_mean[i,:] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['timeLSHNN_mean']
#     timeLSHNN_std[i,:] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['timeLSHNN_std']
#
#     timePaat_mean[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['timePaat_mean']
#     timePaat_std[i] = np.load(folderName + 'stats_prod_'+ str(prod) +'.npz')['timePaat_std']

     
####plotting  
plt.plot( prodList[0:6] , timeLSHNN_mean[0,0:6], color ='red', marker = 'o', label = ' ASSORT-LSH',linewidth=2.0 )
plt.plot( prodList[0:6] , timeExactNN_mean[0,0:6] , color ='blue',  marker = 'o', label = ' ASSORT-EXACT',linewidth=2.0 )
plt.plot( prodList[0:6] , timePaat_mean[0,0:6] , color ='green', marker = 'o', label = 'STATIC-MNL ',linewidth=2.0)
plt.ylabel('Time taken')
plt.xlabel('Number of products')
plt.title('Time taken by different algorithms with max price 1000 and C 50, ncand 200')
plt.legend(loc='best')         
plt.savefig('time_Paat_vs_LSH_ncand 200.png')  
plt.show()


      
plt.errorbar( prodList , timeLSHNN_mean[0,:] , yerr = timeLSHNN_std[0,:] ,linestyle="-",color ='red', marker = 'o', label = ' ASSORT-LSH ',linewidth=2.0)
plt.errorbar( prodList , timeExactNN_mean[0,:] , yerr = timeExactNN_std[0,:] ,linestyle="-",color ='blue', marker = 'o', label = ' ASSORT-EXACT',linewidth=2.0)
plt.ylabel('Time taken')
plt.xlabel('Number of products')
plt.title('Time taken by LSH - finer view, ncand 200')
plt.legend(loc='best')         
plt.savefig('time_LSH_ncand 200.png')  

