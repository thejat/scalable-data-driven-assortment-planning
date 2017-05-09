# -*- coding: utf-8 -*-
"""
Created on Thurs May 05 05:13:14 2017

@author: Theja
"""

import numpy, time
from gurobipy import *

def capAst_LP(prod, C, p, v):
    #v and p are expected to be n+1 and n+1 length lists respectively 
    st = time.time()    

    # Model
    m = Model("capAst")

    items = range(prod+1)

    item = {}
    for i in items:
      item[i] = m.addVar(lb=0,name='item_'+str(i))

    # The objective is to maximize expected revenue
    m.setObjective(sum(item[i]*p[i] for i in items), GRB.MAXIMIZE)

    # constraints
    m.addConstr(sum(item[i] for i in items) == 1, 'sum2one')
    m.addConstr(sum(item[i]*1.0/v[i] for i in items[1:]) - item[0]*C/v[0]<= 0, "capacity")
    for i in items[1:]:
        m.addConstr(item[i]*1.0/v[i] - item[0]*1.0/v[0] <= 0, 'order_'+str(i))

    # Solve
    m.optimize()

    timeTaken = time.time() - st      
    # print " " 
    # print "Results for LP"
    print 'Time taken for running the LP is', timeTaken
    maxRev = 0
    maxRevSet = []
    if m.status == GRB.Status.OPTIMAL:
        maxRev = m.objVal
        # print "Optimal revenue is", maxRev

        itemx = m.getAttr('x', item)
        for i in items[1:]:
            if item[i].x*v[0]/max(v[i],0.00001)/max(item[0].x,0.00001) > 0.0001:
                # print('%s %g' % (i, itemx[i]))
                maxRevSet.append(int(i))

        print "Products in the LP optimal assortment are", maxRevSet 

    else:
        print('No solution')

    return maxRev, set(maxRevSet),timeTaken