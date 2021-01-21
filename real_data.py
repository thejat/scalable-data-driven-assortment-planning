import time
import numpy as np
import random
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import chain
from collections import Counter

def get_feasibles_realdata(fname=None,isCSV=True,min_ast_length=3):
    st = time.time()
    assert fname is not None
   
    with open(fname,'rb') as f:
        data = f.read().split('\n')
        data = data[1:len(data)-1] #data[len(data)-5:len(data)-1] #

    item_ids = set()
    largest_id = 0
    feasibles_raw = []
    C = 0
    for i in data:
		if isCSV==True:
			items_string_spaced = i.split(',')[0]
			# print items_string_spaced
			itemset = [int(x) for x in items_string_spaced.split(' ')]
			# print itemset
		else:
			items_string_spaced = i.split(' #SUP:')[0]
			# print items_string_spaced
			itemset = [int(x) for x in items_string_spaced.split(' ')]
			# print itemset


		if len(itemset) > min_ast_length:
			feasibles_raw.append(itemset)
			for x  in itemset:
				item_ids.add(x)
				if x > largest_id:
					largest_id = x
			C = max(C,len(itemset))

    item_dict = {}
    for e,x in enumerate(item_ids):
		item_dict[x] = e+1

    feasibles = []
        #print  feasibles_raw
    for ast in feasibles_raw:
		set_vector = np.zeros(len(item_ids))
		for x in ast:
			set_vector[item_dict[x]-1] = 1
		feasibles.append(set_vector)
        
    print fname
    print '\tlargest id',largest_id
    print '\tno. unique items',len(item_ids)
    print '\tlen feasibles',len(feasibles)
    print '\tlargest ast size',C
    print "\tloading time: ", time.time()-st


    return feasibles,C,item_ids


def get_feasibles_realdata_by_product(fname=None,isCSV=True,min_ast_length=3, prod = 100, iterNum = 1):
    st = time.time()
    assert fname is not None
   
    with open(fname,'rb') as f:
        data = f.read().split('\n')
        data = data[1:len(data)-1] #data[len(data)-5:len(data)-1] #
   
   	item_ids = set()
   	largest_id = 0
   	feasibles_raw = []  
    C = 0

    for i in data:
   		if isCSV==True:
   			items_string_spaced = i.split(',')[0]
   			# print items_string_spaced
   			itemset = [int(x) for x in items_string_spaced.split(' ')]
   			# print itemset
   		else:
   			items_string_spaced = i.split(' #SUP:')[0]
   			# print items_string_spaced
   			itemset = [int(x) for x in items_string_spaced.split(' ')]
   			# print itemset

   
   		if len(itemset) > min_ast_length:
   			feasibles_raw.append(itemset)
   			for x  in itemset:
   				item_ids.add(x)
   				if x > largest_id:
   					largest_id = x
			C = max(C,len(itemset))
            
    item_dict = {}
    subset = random.sample(item_ids,prod)
    feasible_raw_new=[]
    for e,x in enumerate(subset):
		item_dict[x] = e+1
                
    for j in feasibles_raw:
        if any ([i not in subset for i in j] or False):
            continue
        else:
            feasible_raw_new.append(j)
    
    C = max(len(i) for i in feasible_raw_new) 
    feasibles = []
    for ast in feasible_raw_new:
		set_vector = np.zeros(len(subset))
		for x in ast:
			set_vector[item_dict[x]-1] = 1    
		feasibles.append(set_vector)
   
    
    print fname
    print '\tlargest id',largest_id
    print '\tno. unique items',len(subset)
    print '\tlen feasibles',len(feasibles)
    print '\tlargest ast size',C
    print "\tloading time: ", time.time()-st


    return feasibles,C,subset


def get_feasibles_realdata_by_assortment(fname=None,lenFeas=None, isCSV=True,min_ast_length=3, iterNum = 1):
     st = time.time()
     assert fname is not None
     
     with open(fname,'rb') as f:
          data = f.read().split('\n')
          data = data[1:len(data)-1] #data[len(data)-5:len(data)-1] #
      
     item_ids = set()
     filtered_item_ids = set()
     largest_id = 0
     feasibles_raw = []
     C = 0
     for i in data:
  		if isCSV==True:
  			items_string_spaced = i.split(',')[0]
  			# print items_string_spaced
  			itemset = [int(x) for x in items_string_spaced.split(' ')]
  			# print itemset
  		else:
  			items_string_spaced = i.split(' #SUP:')[0]
  			# print items_string_spaced
  			itemset = [int(x) for x in items_string_spaced.split(' ')]
  			# print itemset
  
  
  		if len(itemset) > min_ast_length:
  			feasibles_raw.append(itemset)
  			for x  in itemset:
  				item_ids.add(x)
  				if x > largest_id:
  					largest_id = x
  			#C = max(C,len(itemset))
  
     choice_indices = np.random.choice(len(feasibles_raw), lenFeas, replace=False)
     choices = [feasibles_raw[i] for i in choice_indices]
     C = len(max(choices, key=len))
     for x in choices:
         for y in x:
  			filtered_item_ids.add(y)
     
     item_dict = {}
     
     for e,x in enumerate(filtered_item_ids):
      		item_dict[x] = e+1
      
     feasibles = []
          #print  feasibles_raw
     for ast in choices:
  		set_vector = np.zeros(len(filtered_item_ids))
  		for x in ast:
  			set_vector[item_dict[x]-1] = 1
  		feasibles.append(set_vector)
          
     print fname
     print '\tlargest id',largest_id
     print '\tno. unique items',len(filtered_item_ids)
     print '\tlen feasibles',len(feasibles)
     print '\tlargest ast size',C
     print "\tloading time: ", time.time()-st
      
      
     return feasibles,C,filtered_item_ids

if __name__=='__main__':

	t0 = time.time()

	#feasibles,C,_ = get_feasibles_realdata('freq_itemset_data/retail0p0001_240852_txns88162.csv',isCSV=True,min_ast_length=3)

	#feasibles,C,_ = get_feasibles_realdata('freq_itemset_data/foodmartFIM0p0001_233231_txns4141.csv',isCSV=True,min_ast_length=4)

	#feasibles,C,_ = get_feasibles_realdata('freq_itemset_data/chains0p00001_txns1112949.txt',isCSV=False,min_ast_length=5)

	#feasibles,C,_ = get_feasibles_realdata('freq_itemset_data/OnlineRetail0p000001_txns540455.txt',isCSV=False,min_ast_length=3)

	#feasibles,C,_ = get_feasibles_realdata('freq_itemset_data/tafeng_v1_0p00001_119578.txt',isCSV=False,min_ast_length=3)

	print "Total loading time:",time.time() - t0    



