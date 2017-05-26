# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:13:14 2017

@author: Theja
"""
import numpy as np
import time
from capAst_oracle import calcRev

def capAst_adxopt(prod, C, p, v, meta = None):

    st = time.time() 
    # initialize
    b = min(C,prod-C+1) #parameter of adxopt, see Thm 3, Jagabathula 
    items = range(1,prod+1)
    removals = np.zeros(prod+1)
    set_prev = []
    rev_prev = calcRev(set_prev, p, v, prod)

    rev_cal_counter = 0
    while True:
        items_left = [x for x in items if x not in set_prev]
        #Additions
        set_addition = []
        rev_addition = 0
        if len(set_prev) < C:
            for j in items_left:
                if removals[j] <b:
                    candidate_rev = calcRev(sorted(set_prev+[j]),p,v,prod)
                    rev_cal_counter +=1
                    if candidate_rev > rev_addition:
                        rev_addition = candidate_rev
                        set_addition = sorted(set_prev+[j])

        #Deletions
        set_deletion = []
        rev_deletion = 0
        if len(set_prev) >0:
            for idx in range(len(set_prev)):
                candidate_rev = calcRev(sorted(set_prev[:idx]+set_prev[idx+1:]),p,v,prod)
                rev_cal_counter +=1
                if candidate_rev > rev_deletion:
                    rev_deletion = candidate_rev
                    set_deletion = sorted(set_prev[:idx]+set_prev[idx+1:])

        #Substitutions
        set_substitution = []
        rev_substitution = 0
        if len(set_prev) >0:
            for j in items_left:
                if removals[j] <b:
                    for idx in range(len(set_prev)):
                        candidate_rev = calcRev(sorted(set_prev[:idx]+[j]+set_prev[idx+1:]),p,v,prod)
                        rev_cal_counter +=1
                        if candidate_rev > rev_substitution:
                            rev_substitution = candidate_rev
                            set_substitution = sorted(set_prev[:idx]+[j]+set_prev[idx+1:])


        idx_rev_current = np.argmax(np.asarray([rev_addition,rev_deletion,rev_substitution]))
        if idx_rev_current==0:
            rev_current = rev_addition
            set_current = set_addition
        elif idx_rev_current==1:
            rev_current = rev_deletion
            set_current = set_deletion
        else:
            rev_current = rev_substitution
            set_current = set_substitution

        if rev_current <= rev_prev or np.min(removals) >= b:
            rev_current = rev_prev
            set_current = set_prev
            break
        else:
            for j in set(set_prev).difference(set(set_current)):
                removals[j] +=1

            rev_prev = rev_current
            set_prev = set_current

    timeTaken = time.time() - st  

    print "\t\tNumber of times calcRev is called:",rev_cal_counter
    print "\t\tProducts in the adxopt assortment are", set_current 
    print '\t\tTime taken for running adxopt is', timeTaken
    
    return rev_current, set_current, timeTaken
        
        