### Code for the paper "Optimizing Revenue over Data-driven Assortments" (2017)

This codebase is for algorithms proposed in the aforementioned [paper](https://arxiv.org/abs/1708.05510). It consists of the following python scripts:

  * proposed_algos.py : has the proposed algorithms
  * competing_algos.py : has the competitor algorithms 
  * real_data.py : processes data (e.g., from the billion prices project or the frequent itemset data)
  * all_algos_test.py : calls algorithms on various instances
  * plots_paper.py : generates plots given in the paper

The easiest way to get started is to look at all_algos_test.py and go from there. 

##### Additional Notes 

 * We use Gurobi/Cplex for solving the competing linear program. These can be obtained from the respective websites.
 * The datasets (billion prices and frequent itemsets) are not supplied here. Please download from the respective websites (see paper). Some minimal preprocessing is needed. We recommend starting with synthetic instances at the very outset.


Please make a pull request if you spot bugs or have suggestions!