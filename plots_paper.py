import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
plt.rcParams.update({'legend.fontsize': 'xx-large',
    'axes.labelsize': '24',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large',
    'figure.autolayout': True,
    'text.usetex'      : False})


def get_plot_subroutine(params):
    # plt.gca().cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = params['loggs']['additional'][params['xsname']]
    algonames_new = params['loggs']['additional']['algonames']#['Assort-MNL(approx)','Assort-MNL','Adxopt','LP']

    for e,algo in enumerate(params['loggs']['additional']['algonames']):
        if params['flag_rmadxopt']==True and algo=='Adxopt':
            continue
        else:
            if params['flag_bars']==True:
                ys_lb  = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],25) for i in range(len(xs))])
                ys_ub  = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],75) for i in range(len(xs))])
                ax.fill_between(xs, ys_lb, ys_ub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            # ys = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],50) for i in range(len(xs))])
            ys = np.asarray([np.mean(params['loggs'][algo][params['logname']][i,:]) for i in range(len(xs))])
            if algo=='Adxopt':
                ax.plot(xs, ys,label=algo,marker='>',markersize=10)
            elif algo=='LP':
                ax.plot(xs, ys,label=algo,marker='<',markersize=10)
            else:
                ax.plot(xs, ys,label=algo)
        print algo, algonames_new[e],ys


    ax.legend(loc='best', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.ylabel(params['ylab'])
    plt.xlabel(params['xlab'])
    plt.legend(loc='best')
    plt.xlim(params['xlims'])
    if params['ylims'] is not None:
        plt.ylim(params['ylims'])

    if params['flag_savefig'] == True:
        plt.savefig(params['fname'])  
    plt.show()

def get_plots(fname=None,flag_savefig=False,xlim=5001,loggs=None,
    xsname='prodList',xlab='Number of Products',savefname_common='./output/undefined'):

    #Load data
    if fname is None and loggs is None:
        print 'No data or pickle filename supplied.'
        return 0
    elif fname is not None:
        loggs = pickle.load(open(fname,'rb'))
    else:
        fname = 'undefined0000'#generic

    ####plot1
    params = {'fname':savefname_common+'_time.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs,'flag_bars':False,'xlab':xlab,'ylab':'Time (s)',
        'logname':'time','xsname':xsname,'ylims':None,'flag_rmadxopt':False}
    get_plot_subroutine(params)


    ###plot2
    params = {'fname':savefname_common+'_revPctErr.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs,'flag_bars':False,'xlab':xlab,'ylab':'Pct. Err. in Revenue',
        'logname':'revPctErr','xsname':xsname,'ylims':[-.02,0.3],'flag_rmadxopt':False}
    get_plot_subroutine(params)


    ###plot3
    params = {'fname':savefname_common+'_setOlp.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs,'flag_bars':False,'xlab':xlab,'ylab':'Pct. Set Overlap',
        'logname':'setOlp','xsname':xsname,'ylims':[0,1.1],'flag_rmadxopt':False}
    get_plot_subroutine(params)

def get_freqitem_subroutine(params):

    loggs = params['loggs']
    loggs['additional']['algonames_new'] = ['Assort-MNL(approx)','Assort-MNL','Exhaustive']
    ind = np.arange(len(loggs['additional']['real_data_list']))  # the x locations for the groups
    width = 0.25       # the width of the bars
    fig, ax = plt.subplots()

    rects = {}
    colors = 'rbgkymc'
    for e,algoname in enumerate(loggs['additional']['algonames']):
        rects[algoname] = ax.bar(ind+e*width,tuple(np.mean(loggs[algoname][params['logname']][i,:]) for i in ind), width, color=colors[e])

    ax.set_ylabel(params['ylab'])
    if params['ylim'] is not None:
        ax.set_ylim(params['ylim'])
    ax.set_xticks(ind+1.4*width)
    ax.set_xticklabels( ('Retail', 'Foodmart', 'Chainstore', 'E-commerce') )
    ax.legend( (rects[algoname][0] for algoname in loggs['additional']['algonames']), tuple(loggs['additional']['algonames_new']),loc='best')

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'%(height),
                    ha='center', va='bottom')

    for algoname in loggs['additional']['algonames']:
        autolabel(rects[algoname])
    

    if params['flag_savefig'] == True:
        plt.savefig(params['fname']) 
    # plt.show()

def get_freqitem_plots(fname,flag_savefig=False):

    #Load data
    if fname is None and loggs is None:
        print 'No data or pickle filename supplied.'
        return 0
    elif fname is not None:
        loggs = pickle.load(open(fname,'rb'))
    else:
        fname = 'undefined0000'#generic

    ####plot1 #fname[:-4]+'_time.png'
    params = {'fname':'./output/figures/gen_ast_real_time.png','flag_savefig':flag_savefig,
        'loggs':loggs,'ylab':'Time (s)','logname':'time','ylim':None}
    get_freqitem_subroutine(params)


    ###plot2 #fname[:-4]+'_revPctErr.png'
    params = {'fname':'./output/figures/gen_ast_real_revPctErr.png','flag_savefig':flag_savefig,
        'loggs':loggs,'ylab':'Pct. Err. in Revenue','logname':'revPctErr','ylim':[0,.1]}
    get_freqitem_subroutine(params)


    ###plot3 #fname[:-4]+'_setOlp.png'
    params = {'fname':'./output/figures/gen_ast_real_setOlp.png','flag_savefig':flag_savefig,
        'loggs':loggs,'ylab':'Pct. Set Overlap','logname':'setOlp','ylim':[0,1.7]}
    get_freqitem_subroutine(params)



def get_merged_plot_subroutine(params):
    # plt.gca().cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = params['loggs']['additional'][params['xsname']]
    algonames_new = params['algonames']

    for e,algo in enumerate(params['loggs']['additional']['algonames']):
        if params['flag_rmadxopt']==True and algo=='Adxopt':
            continue
        else:
            if params['flag_bars']==True:
                ys_lb  = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],25) for i in range(len(xs))])
                ys_ub  = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],75) for i in range(len(xs))])
                ax.fill_between(xs, ys_lb, ys_ub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            # ys = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],50) for i in range(len(xs))])
            ys = np.asarray([np.mean(params['loggs'][algo][params['logname']][i,:]) for i in range(len(xs))])
            if algo=='Adxopt':
                ax.plot(xs, ys,label=algo,marker='>',markersize=10)
            elif algo=='LP':
                ax.plot(xs, ys,label=algo,marker='<',markersize=10)
            else:
                ax.plot(xs, ys,label=algo)
        print algo, algonames_new[e],ys


    ax.legend(loc='best', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.ylabel(params['ylab'])
    plt.xlabel(params['xlab'])
    plt.legend(loc='best')
    plt.xlim(params['xlims'])
    if params['ylims'] is not None:
        plt.ylim(params['ylims'])

    if params['flag_savefig'] == True:
        plt.savefig(params['fname'])  
    plt.show()



def get_merged_plots(fname_prefix,fnames,flag_savefig,nest_ncand,xlim)
    xsname,xlab = 'lenFeasibles','Number of Assortments'

    alldata = {}
    for e,fname in enumerate(fnames):
        alldata[nest_ncand[e]] = pickle.load(open(fname_prefix+fname,'rb'))

    algonames1 = ['Exhaustive']
    algonames2 = ['Assort-MNL(approx)','Assort-MNL']
    params[algoname1] = 1


if __name__ == '__main__':

    #1. general: 
    #bpp 
    fname_prefix = './output/results20170526_final_vs_nassortments/'
    fnames = ['gen_loggs_bppData_lenF_51200_nCand_80_nEst_20_20170527_0746PM.pkl',
    'gen_loggs_bppData_lenF_51200_nCand_160_nEst_40_20170526_1124AM.pkl',
    'gen_loggs_bppData_lenF_51200_nCand_200_nEst_100_20170526_0118PM.pkl']
    nest_ncand = [(20,80),(40,160),(100,200)]
    get_merged_plots(fname_prefix,fnames,flag_savefig=False,nest_ncand=nest_ncand,xlim=51201)

    #synthetic
    # fname = './output/results20170526_final_vs_nassortments/gen_loggs_synthetic_lenF_51200_nCand_80_nEst_20_20170527_1043PM.pkl'
    # get_plots(fname,flag_savefig=False,xlim=xlim,xsname=xsname,xlab=xlab)





    #2. DONE general: freq_item_dataset
    # get_freqitem_plots('./output/results20170527_final_freq_sets/gen_loggs_real_ast_upto3_nCand_160_nEst_40_20170528_0154AM.pkl',flag_savefig=True)




    #3. REDO capacity constrained: 
    # #bpp
    # xlim,fname = 10001,'./output/results20170527_final_vs_prod/cap_loggs_bppData_prod_10000_20170527_1255AM.pkl'
    # get_plots(fname=fname,flag_savefig=True,xlim=xlim,
    #     savefname_common='./output/figures/cap_realprice_prod')
    # #synthetic
    # xlim,fname = 20001,'./output/results20170527_final_vs_prod/cap_loggs_synthetic_prod_20000_20170526_1016PM.pkl'
    # get_plots(fname=fname,flag_savefig=True,xlim=xlim,
    #     savefname_common='./output/figures/cap_synthetic_prod')


