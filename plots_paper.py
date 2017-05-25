import numpy as np
import matplotlib.pyplot as plt
import pickle


def get_plots(fname=None,flag_savefig=False,xlim=5001,loggs=None):

    #Plotting parameters
    params = {'axes.labelsize': 'x-large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)


    #Load data
    if fname is None and loggs is None:
        print 'No data or pickle filename supplied.'
        return 0
    elif fname is not None:
        loggs = pickle.load(open(fname,'rb'))
    prodList = loggs['additional']['prodList']
    algonames = loggs['additional']['algonames']
         
    ####plot1
    for algo in algonames:
        if algo=='paat':
            continue
        plt.errorbar( prodList, loggs[algo]['time_mean'],
        yerr = loggs[algo]['time_std'], 
        linestyle="-", marker = 'o', label = algo, linewidth=2.0)
    plt.ylabel('Time (s)')
    plt.xlabel('Number of products')
    plt.title('Computational Performance')
    plt.legend(loc='best')
    plt.xlim([0,xlim])
    if flag_savefig == True:
        plt.savefig(fname[:-4]+'_time.png')  
    plt.show()


    ###plot2
    for algo in algonames:
        plt.errorbar( prodList, loggs[algo]['revPctErr_mean'],
        yerr = loggs[algo]['revPctErr_std'], 
        linestyle="-", marker = 'o', label = algo, linewidth=2.0)
    plt.ylabel('Revenue Pct Err')
    plt.xlabel('Number of products')
    plt.title('Approx. Quality 1')
    plt.legend(loc='best')
    # plt.ylim([-.1,1.1])
    plt.xlim([0,xlim])
    if flag_savefig == True:
        plt.savefig(fname[:-4]+'_revPctErr.png')  
    plt.show()

    ###plot3
    for algo in algonames:
        plt.plot( prodList, loggs[algo]['setOlp_mean'],
        linestyle="-", marker = 'o', label = algo, linewidth=2.0)
    plt.ylabel('Set Overlap')
    plt.xlabel('Number of products')
    plt.title('Approx. Quality 2')
    plt.legend(loc='best')
    # plt.ylim([-.1,1.1])
    plt.xlim([0,xlim])
    if flag_savefig == True:
        plt.savefig(fname[:-4]+'_setOlp.png')  
    plt.show()

    ###plot4
    for algo in algonames:
        plt.plot( prodList, loggs[algo]['corrSet_mean'],
        linestyle="-", marker = 'o', label = algo, linewidth=2.0)
    plt.ylabel('Set Overlap Normalized')
    plt.xlabel('Number of products')
    plt.title('Approx. Quality 3')
    plt.legend(loc='best')
    # plt.ylim([-.1,1.1])
    plt.xlim([0,xlim])
    if flag_savefig == True:
        plt.savefig(fname[:-4]+'_corrSet.png')  
    plt.show()


if __name__ == '__main__':

    fname = './output/loggs_synthetic_200_20170524_0355AM.pkl'
    xlim = 5001

    get_plots(fname,flag_savefig=False,xlim=xlim)