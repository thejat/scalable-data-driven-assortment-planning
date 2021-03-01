import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import pickle
from math import log
sns.set()



plt.rcParams.update({'legend.fontsize': 'xx-large',
    'axes.labelsize': '24',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large',
    'figure.autolayout': True,
    'text.usetex'      : False})


def get_plot_subroutine(params):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = params['loggs']['additional'][params['xsname']]
    algonames_new = ['Assort-MNL(approx)','Assort-MNL','Adxopt','LP'] # params['loggs']['additional']['algonames']
    # print algonames_new

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
            elif algo=='LP' and params['logname'] != 'time':
                ax.plot(xs, ys,label=algo,marker='<',markersize=10)
            else:
                ax.plot(xs, ys,label=algo)
        # print algo, algonames_new[e],ys


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

def get_adx_plot(params):

    #setting auxiliary stuff
    cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

            'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0)),  # no green at 1

            'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                      (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0)) }  # no blue at 1
    GnRd = colors.LinearSegmentedColormap('GnRd', cdict)


    #adxopt data {'fname':savefname_common+'adxopt_time.png','flag_savefig':flag_savefig,'xlims':[0,xlim],'loggs':loggs}
    opt_ast_lens = np.zeros((len(params['loggs']['additional']['prodList']),params['loggs']['additional']['N']))
    for i,_ in enumerate(params['loggs']['additional']['prodList'][:]):
        for j in range(params['loggs']['additional']['N']):
            opt_ast_lens[i,j] = len(params['loggs']['Adxopt']['maxSet'][(i,j)])
    
    print params['loggs']['additional']['prodList'][:-5]
    print opt_ast_lens    

    timedata = params['loggs']['Adxopt']['time']

    n_optast_bucket = 4 #0-25,25-50,50-75,75-100
    thresholds = [0]
    for i in [25,50,75,100]:
        thresholds.append(np.percentile(opt_ast_lens.flatten(),i))
    thresholds = [0,5,10,20,50]
    def threshold2idx(x,thresholds):
        for i in range(len(thresholds)-1):
            if x> thresholds[i] and x<=thresholds[i+1]:
                # print 'threhsold index is',i
                return i

    prodList = params['loggs']['additional']['prodList'][:-5]
    data = np.zeros((n_optast_bucket,len(prodList)))
    count = np.copy(data)
    for i,_ in enumerate(prodList):
        for j in range(len(timedata[i])):
            k = threshold2idx(opt_ast_lens[i,j],thresholds)
            # print 'i,j,k',i,j,k
            data[k,i]  += timedata[i,j]
            count[k,i] +=1
    data = data/count
    # print data,count

    ##debugging
    # for i,prod in enumerate(prodList):
    #     for j in range(len(timedata[i])):
    #         if opt_ast_lens[i,j]>30:
    #             print 'prod size',prod,'ast size:',opt_ast_lens[i,j],'time taken',timedata[i,j]


    fig,ax = plt.subplots(1)
    p=ax.pcolor(data,cmap='RdBu',vmin=np.nanmin(data),vmax=np.nanmax(data))
    # p=ax.pcolor(count,cmap='RdBu',vmin=np.nanmin(count),vmax=np.nanmax(count))
    clt = fig.colorbar(p,ax=ax)
    clt.ax.get_yaxis().labelpad = 25
    clt.ax.set_ylabel('Time(s)', rotation=270)

    # Set the ticks and labels...
    plt.xticks(range(len(prodList)),prodList)
    plt.xlabel('Number of Items')
    ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.yticks(range(len(thresholds)),thresholds)
    plt.ylabel('Optimal assortment size')

    if params['flag_savefig'] == True:
        plt.savefig(params['fname'])  
    plt.show()
    return timedata,opt_ast_lens,data

#################################################################

def get_plots_temp(fname,flag_savefig=False,xlim=5001,loggs=None,
    xsname='prodList',xlab='Number of Items',savefname_common='./output/undefined'):

    #Load data
    loggs = pickle.load(open(fname,'rb'))

    ###plot1
    params = {'fname':savefname_common+'_time.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs,'flag_bars':False,'xlab':xlab,'ylab':'Time (s)',
        'logname':'time','xsname':xsname,'ylims':None,'flag_rmadxopt':True}
    get_plot_subroutine_temp(params)


    ###plot2
    params = {'fname':savefname_common+'_revPctErr.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs,'flag_bars':False,'xlab':xlab,'ylab':'Pct. Err. in Revenue',
        'logname':'revPctErr','xsname':xsname,'ylims':[0.00,0.06],'flag_rmadxopt':False}
    get_plot_subroutine_temp(params)


    ###plot3
    params = {'fname':savefname_common+'_setOlp.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs,'flag_bars':False,'xlab':xlab,'ylab':'Pct. Set Overlap',
        'logname':'setOlp','xsname':xsname,'ylims':[0,1.1],'flag_rmadxopt':False}
    get_plot_subroutine_temp(params)

    return 0,0,0

def get_plot_subroutine_temp(params):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = params['loggs']['additional'][params['xsname']]
    algonames_new = {'Assort-LSH':'Assort-MNL(approx)','Assort-Exact':'Assort-MNL','Adxopt':'ADXOpt','LP':'LP'} # params['loggs']['additional']['algonames']
    # print algonames_new
    params['loggs']['additional']['algonames'] = ['Adxopt', 'Assort-Exact',  'LP']
    #print(params['loggs']['additional']['algonames'])
    for e,algo in enumerate(params['loggs']['additional']['algonames']):
        if params['flag_rmadxopt']==True and algo=='Adx-opt':
            continue
        else:
            if params['flag_bars']==True:
                ys_lb  = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],25) for i in range(len(xs))])
                ys_ub  = np.asarray([np.percentile(params['loggs'][algo][params['logname']][i,:],75) for i in range(len(xs))])
                ax.fill_between(xs, ys_lb, ys_ub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            ys = np.asarray([np.mean(params['loggs'][algo][params['logname']][i,:]) for i in range(len(xs))])    
            #ys = params['loggs'][algo][params['logname']]    
            if algo =='Adxopt':
                ax.set_yscale('log')
                ax.plot(xs, ys,label=algonames_new[algo], marker = '<', markersize=10, color = 'orange')
            elif algo=='LP':
                ax.plot(xs, ys,label=algonames_new[algo], marker = '>', markersize=10, color = 'k', alpha =0.8)
            else: 
                ax.plot(xs, ys,label=algonames_new[algo], marker = '*', markersize=10, color = 'g')
        # print algo, algonames_new[e],ys

    ax.set_facecolor('white')
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('k')
    import matplotlib.ticker as mtick
    #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))      
    plt.rcParams["figure.figsize"] = (8,6)
    ax.legend(loc='best', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.ylabel(params['ylab'])
    plt.xlabel(params['xlab'])
    plt.legend(loc='best', facecolor= 'inherit', framealpha=0.0, frameon=None)
    plt.xlim(params['xlims'])
    if params['ylims'] is not None:
        plt.ylim(params['ylims'])
    plt.xticks(np.arange(0, 15001, 2500))
    if params['flag_savefig'] == True:
        plt.savefig(params['fname'])  
    plt.show()
    

#############################################

def get_plots(fname,flag_savefig=False,xlim=5001,loggs=None,
    xsname='prodList',xlab='Number of Items',savefname_common='./output/undefined'):

    #Load data
    loggs = pickle.load(open(fname,'rb'))

    ###plot0
    params = {'fname':savefname_common+'_adxopt.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs}
    timedata,opt_ast_lens,data= get_adx_plot(params)
    # return timedata,opt_ast_lens,data

    ###plot1
    params = {'fname':savefname_common+'_time.png','flag_savefig':flag_savefig,'xlims':[0,xlim],
        'loggs':loggs,'flag_bars':False,'xlab':xlab,'ylab':'Time (s)',
        'logname':'time','xsname':xsname,'ylims':None,'flag_rmadxopt':True}
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

    return 0,0,0

def get_freqitem_subroutine(params):

    loggs = params['loggs']
    loggs['additional']['algonames'] = ['Assort-LSH-G', 'Assort-BZ-G', 'Assort-Exact-G', 'Linear-Search']
    #loggs['additional']['algonames_new'] = ['Assort-MNL(approx)','Assort-MNL','Exhaustive','Assort-MNL(BZ)']\
    loggs['additional']['algonames_new'] = ['Assort-MNL(Approx)','Assort-MNL(BZ)','Assort-MNL','Exhaustive']
    #loggs['additional']['algonames_new'] = ['Exhaustive','Assort-MNL(BZ)']
    ind = np.arange(len(loggs['additional']['real_data_list']))  # the x locations for the groups
    width = 0.2       # the width of the bars
    fig, ax = plt.subplots()

    rects = {}
    colors = 'rbgkymc'
    for e,algoname in enumerate(loggs['additional']['algonames']):
        rects[algoname] = ax.bar(ind+e*width,tuple(np.mean(loggs[algoname][params['logname']][i,:]) for i in ind), width, color=colors[e])
    
    ax.set_ylabel(params['ylab'])
    if params['ylim'] is not None:
        ax.set_ylim(params['ylim'])
    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels( ('Retail', 'Foodmart', 'Chainstore', 'E-commerce','Tafeng') )
    ax.set_facecolor('white')
    ax.legend( (rects[algoname][0] for algoname in loggs['additional']['algonames']), tuple(loggs['additional']['algonames_new']),loc='best', framealpha=0.0, frameon=None)
    plt.rcParams["figure.figsize"] = (12,8)
    import matplotlib.ticker as mtick

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            #ax.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%.2f'%(height),
            ax.text(rect.get_x()+rect.get_width()/2., 1.02*height, "{:.0%}".format(height),        
                    
                    ha='center', va='bottom', fontsize = 15)

    for algoname in loggs['additional']['algonames']:
        autolabel(rects[algoname])
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('k')
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
    params = {'fname':'./output/figures/gen_ast_real_set_time.png','flag_savefig':flag_savefig,
        'loggs':loggs,'ylab':'Time (s)','logname':'time','ylim':None}
    get_freqitem_subroutine(params)


    ###plot2 #fname[:-4]+'_revPctErr.png'
    params = {'fname':'./output/figures/gen_ast_real_set_revPctErr.png','flag_savefig':flag_savefig,
        'loggs':loggs,'ylab':'Pct. Err. in Revenue','logname':'revPctErr','ylim':[0,.2]}
    get_freqitem_subroutine(params)


    ###plot3 #fname[:-4]+'_setOlp.png'
    params = {'fname':'./output/figures/gen_ast_real_set_setOlp.png','flag_savefig':flag_savefig,
        'loggs':loggs,'ylab':'Pct. Set Overlap','logname':'setOlp','ylim':[0,1.7]}
    get_freqitem_subroutine(params)

def get_merged_plot_subroutine(params):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = params['xs']
    params['algonames'] = ['Assort-MNL(Approx)', 'Assort-MNL(BZ)', 'Assort-MNL', 'Exhaustive']
    colors = 'rbgkymc'
    
    print(params['algonames'])
    for e,algo in enumerate(params['algonames']):
        ys = params[algo][params['logname']]
        ax.plot(xs, ys,label=algo, color = colors[e])
    plt.rcParams["figure.figsize"] = (8,6)    
    ax.legend(loc='best', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.ylabel(params['ylab'][params['logname']])
    plt.xlabel(params['xlab'])
    plt.legend(loc='best', facecolor= 'inherit', framealpha=0.0, frameon=None)
    plt.xlim(params['xlims'])
    if params['ylims'][params['logname']] is not None:
        plt.ylim(params['ylims'][params['logname']])
    if params['flag_savefig'] == True:
        plt.savefig(params['fname'])  
    ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_facecolor('white')
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('k')
    import matplotlib.ticker as mtick
    #plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))    
    plt.show()

    

def get_merged_plots(fname_prefix,fnames,flag_savefig,nest_ncand,xlim,dat):

    #read data
    alldata = {}
    for e,fname in enumerate(fnames):
        alldata[nest_ncand[e]] = pickle.load(open(fname_prefix+fname,'rb'))
    # print alldata[nest_ncand[0]].keys()
    lognames = ['time','revPctErr','setOlp']

    params = {}
    params['flag_savefig'] = flag_savefig
    params['xs']   = alldata[nest_ncand[0]]['additional']['lenFeasibles']
    params['xlab'] = 'Number of Assortments'
    params['xlims'] = [0,xlim]
    params['algonames'] = ['Exhaustive','Assort-MNL']
    params['Exhaustive'] = {}
    params['Assort-MNL'] = {}
    for logname in lognames:
        params['Exhaustive'][logname] = np.asarray([np.mean(alldata[nest_ncand[0]]['Linear-Search'][logname][i,:]) for i in range(len(params['xs']))])
        params['Assort-MNL'][logname] = np.asarray([np.mean(alldata[nest_ncand[0]]['Assort-Exact-G'][logname][i,:]) for i in range(len(params['xs']))])

    algoTemp = 'Assort-MNL(Approx)'
    for nest,ncand in nest_ncand:
        #algoTempNew = algoTemp+'['+str(nest)+','+str(ncand)+']'
        algoTempNew = algoTemp
        params['algonames'].append(algoTempNew)
        params[algoTempNew] = {}
        for logname in lognames:
            params[algoTempNew][logname] = np.asarray([np.mean(alldata[(nest,ncand)]['Assort-LSH-G'][logname][i,:]) for i in range(len(params['xs']))])
    
    algoTemp = 'Assort-MNL(BZ)'
    for nest,ncand in nest_ncand:
        if nest == 20 and ncand == 80:
            #algoTempNew = algoTemp+'['+str(nest)+','+str(ncand)+']'
            algoTempNew = algoTemp
            params['algonames'].append(algoTempNew)
            params[algoTempNew] = {}
            for logname in lognames:
                params[algoTempNew][logname] = np.asarray([np.mean(alldata[(nest,ncand)]['Assort-BZ-G'][logname][i,:]) for i in range(len(params['xs']))])        

    params['ylab'] = {'time':'Time (s)', 'revPctErr':'Pct. Err. in Revenue','setOlp':'Pct. Set Overlap' }
    params['ylims'] = {'time':None, 'revPctErr':[0.0,0.2],'setOlp':[0,1.1] }
    for logname in lognames:
        params['logname'] = logname
        params['fname'] = './output/figures/gen_ast_'+dat+'_price_'+logname+'.png'
        get_merged_plot_subroutine(params)


def get_static_mnl_plot(fname, fname1,flag_savefig,xlim,savefname):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = 'gr'
    count = 0
    for i in [fname, fname1]:
        print(i)
        loggs = pickle.load(open(fname,'rb')) 
        xs = loggs['additional']['prodList']
        ys_lb  = np.asarray([np.percentile(loggs['Static-MNL']['time'][i,:],5) for i in range(len(xs))])
        ys_ub  = np.asarray([np.percentile(loggs['Static-MNL']['time'][i,:],95) for i in range(len(xs))])
        ax.fill_between(xs, ys_lb, ys_ub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        ys = np.asarray([np.mean(loggs['Static-MNL']['time'][i,:]) for i in range(len(xs))])
        if count == 0:
            ax.plot(xs, ys,label='Static-MNL(C=50)', color = 'b', marker = '', markersize=10)
        if count == 1:
            ax.plot(xs, ys,label='Static-MNL(C=100)', color = 'b', marker = 'o', markersize=10)

        count = count + 1
        
        
    plt.rcParams["figure.figsize"] = (8,6)
    ax.legend(loc='best', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.ylabel('Time (s)')
    plt.xlabel('Number of Items')
    plt.legend(loc='best', facecolor= 'inherit', framealpha=0.0, frameon=None)
    plt.xlim([0,xlim])
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('k')
    ax.set_facecolor('white')
    if flag_savefig == True:
        plt.savefig(savefname)  
    # plt.show()

if __name__ == '__main__':

    ##1. DONE general as a function of assortment set size: 
    # #bpp 
    # fname_prefix = './output/results20170526_final_vs_nassortments/'
    # fnames = ['gen_loggs_bppData_lenF_51200_nCand_80_nEst_20_20170527_0746PM.pkl',
    # 'gen_loggs_bppData_lenF_51200_nCand_160_nEst_40_20170526_1124AM.pkl',
    # 'gen_loggs_bppData_lenF_51200_nCand_200_nEst_100_20170526_0118PM.pkl']
    # nest_ncand = [(20,80),(40,160),(100,200)]
    # get_merged_plots(fname_prefix,fnames,flag_savefig=True,nest_ncand=nest_ncand,xlim=51201,dat='real')
    
    
     #bpp 
    # fname_prefix = './output/results_final_vs_nassortments/'
    # fnames = ['gen_loggs_bppData_lenF_51200_nCand_80_nEst_20_20210227_0426PM.pkl']
    # nest_ncand = [(20,80)]
    # get_merged_plots(fname_prefix,fnames,flag_savefig=True,nest_ncand=nest_ncand,xlim=51201,dat='real')
    

    # #synthetic
# =============================================================================
#      fname_prefix = './output/results_final_vs_nassortments/'
#      fnames = ['gen_loggs_tafeng_lenF_51200_nCand_80_nEst_20_20201228_0619PM.pkl',
#      'gen_loggs_tafeng_lenF_51200_nCand_160_nEst_40_20201228_0140PM.pkl',
#      'gen_loggs_tafeng_lenF_51200_nCand_200_nEst_100_20201228_0417AM.pkl']
#      nest_ncand = [(20,80),(40,160),(100,200)]
#      get_merged_plots(fname_prefix,fnames,flag_savefig=True,nest_ncand=nest_ncand,xlim=51201,dat='synthetic')
# =============================================================================


    ##2. DONE general: freq_item_dataset
    # get_freqitem_plots('./output/trial_for_ast_utility.pkl',flag_savefig = True)
    # get_freqitem_plots('./output/gen_loggs_real_ast_upto3_nCand_80_nEst_20_20210108_0151AM_compstep_0p9_hash2.pkl',flag_savefig=True)



    ##3. DONE capacity constrained: 
    # # #bpp
    # xlim,fname = 20001,'./output/results20170528_final_vs_prod/cap_loggs_bppData_prod_20000_20170529_0459AM.pkl'
    # timedata,opt_ast_lens,data = get_plots_temp(fname=fname,flag_savefig=True,xlim=xlim,
    #     savefname_common='./output/figures/new/cap_real_price_prod')

    # xlim,fname,fname1 = 1001,'./output/cap_loggs_tafeng_prod_1000_20210226_0828PM_cap50_staticmnl_tafeng.pkl', './output/cap_loggs_tafeng_prod_1000_20210226_1035PM_cap100_tafeng_staticmnl.pkl'
    # get_static_mnl_plot(fname=fname,fname1 = fname1, flag_savefig=True,xlim=xlim,savefname='./output/figures/cap_real_price_prod_staticmnl.png')

    # #synthetic
    # xlim,fname = 20001,'./output/results20170528_final_vs_prod/cap_loggs_synthetic_prod_20000_20170529_1214AM.pkl'
    # timedata,opt_ast_lens,data = get_plots(fname=fname,flag_savefig=True,xlim=xlim,
    #     savefname_common='./output/figures/cap_synthetic_prod')

    # xlim,fname = 10001,'./output/cap_loggs_synthetic_prod_10000_20180416_0315PM.pkl'
    # timedata,opt_ast_lens,data = get_plots(fname=fname,flag_savefig=True,xlim=xlim,
    #    savefname_common='./output/cap_synthetic_prod_linsacan')
    
    #xlim,fname = 15001,'./output/try_me.pkl'
    #timedata,opt_ast_lens,data = get_plots_temp(fname=fname,flag_savefig=True,xlim=xlim,
    #    savefname_common='./output/figures/new/cap_real_price_prod')
