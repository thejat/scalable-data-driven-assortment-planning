# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 08:43:28 2017

@author: Deeksha
"""

import os
import numpy as np

price_range = 20000

fileName = os.getcwd() + '/billion_price_data/processed_data/usa_5/prices_'
fileNameList = []
for day in range(10,31):
    for month in ['jun', 'jul', 'aug']:
        chosenDay = str(day) + month + '2009'
        fileNameList.append(fileName+ chosenDay+'.npy')
        
valid_size = np.zeros(len(fileNameList))   
orig_size = np.zeros(len(fileNameList))      
i = 0        
for files in fileNameList:   
     allPrices = np.load(files)
     allValidPrices = allPrices[allPrices < price_range]
     valid_size[i] = len(allValidPrices)
     orig_size[i] = len(allPrices)
     i = i+1
     
     
                
            