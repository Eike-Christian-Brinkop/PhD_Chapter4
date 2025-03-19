# -*- coding: utf-8 -*-
"""
This published version was created on 19.03.2025

@author: Eike-Christian Brinkop
https://www.linkedin.com/in/eike-christian-brinkop-138158211/


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~  Please read this documentation before usage  ~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This script is used for the performance measurement of the results.

Please ensure that you
- replace BASE_PATH with the path on you machine
- ensure that the folder structure afterwards contains the following folders:
    - Data_P3/
        - f500
        - complete
    - Paper3 results
        - Iterations
        - Processed
            -date
        - Models
        - Timers
        

"""

from utils import print_hint2, QAA
import os, pandas as pd, numpy as np, scipy as sc, time, threading as th, copy, \
    math, seaborn as sns, torch as to
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.api as smapi
import matplotlib.colors as mcolors

VARIANT = "f500"
YEARS = range(2003,2023)

BASE_PATH = "D:/Backup Research/"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "C:/BT/"
PATH_DATA_RAW = BASE_PATH + "Data_P3/"
 
extra_path = "" #"Archive2509/"

variant = "f500"
PATH_PREDICTIONS    = BASE_PATH+'Paper3 results/Iterations/{extra_path}{model_name}_{objective_str}_{VARIANT}_pred{year}.h5'
PATH_PROCESSED      = BASE_PATH+ "Paper3 results/Processed/{date:s}/{model_name}_{VARIANT}_{name_type}.{fileending}"
PATH_DATA           = BASE_PATH+ f"Data_P3/{variant}/"

PROCESSED_PATH = BASE_PATH + "Paper3 results/Processed/"
MODELS_PATH = BASE_PATH + "Paper3 results/Models/"
PATH_FF5_file = "C:/BT/Data/FFportfolios.h5"

COLUMNS_RESULTS = ["res_naive"]
DATE_ENTITY = "date"
IDENTIFIER = "gvkey"

COLUMNS_PORTFOLIOS_DEFAULT = ["managed","HML_vv", 
                               "high_vv","low_vv",
                              "HML_ev",
                               "high_ev","low_ev"
                              ]
COLUMNS_PORTFOLIOS_TABLES = [
    "managed","HML_vv", "HML_ev", # "high_vv","low_vv", "high_ev","low_ev"
    ]



# models_FFNN_w = [
#     {"name":"RP","color":"red"},
#     {"name":"SDF","color":"blue"},
#     {"name":"VP","color":"green"},
#     {"name":"VP_HARQ","color":"orange"},
#     ]

# models_other = [
#     {"name":"RP","color":"red"},
#     {"name":"SDF","color":"blue"},
#     {"name":"VP","color":"green"},
#     ]

models = [
    {"name":"FFNN_w","color":"red","ls":"dotted"},
    {"name":"FFNN_wmq_joined","color":"blue","ls":"dotted"},
    {"name":"FFNN_wmq_mixed","color":"green","ls":"dotted"},
    {"name":"LB_w","color":"red","ls":"solid"},
    {"name":"LB_wmq_joined","color":"blue","ls":"solid"},
    # {"name":"LB_wmq_mixed","color":"green","ls":"solid"},
    ]


"""
create mcap dataset:
    
"""
def get_series(
        columns="mcap", 
        frequency = "w",
        path = PATH_DATA,
        create_file = True):
    if type(columns) is str:
        columns = [columns]
    
    if create_file and os.path.exists(path+"_".join(columns)+".h5"):
        files_loaded = pd.read_hdf(path+"_".join(columns)+".h5",key="data")
    else:
        files = [file for file in os.listdir(path) if file[-5:] == f"x{frequency}.h5"]
        files_loaded = []
        for file in files:
            files_loaded.append(
                pd.read_hdf(path+file,key="data")[columns]
                )
        files_loaded = pd.concat(files_loaded)
        if create_file:
            files_loaded.to_hdf(path+"_".join(columns)+".h5",key="data")
    return files_loaded

def loader_regressions_P3(
        model_name:str = "FFNN_w",
        objective_str:str = "VP",
        years:list = YEARS,
        path:str = PATH_PREDICTIONS
        ):
    '''
    Loads (and concatenates) prediction results

    Parameters
    ----------
    model_name : TYPE
        DESCRIPTION.
    years : TYPE
        DESCRIPTION.
    path : TYPE, optional
        DESCRIPTION. The default is PATH_RETURNS.

    Returns
    -------
    files : TYPE
        DataFrame of predictions.

    '''
    # model_name = "FFNN_w"; objective_str="VP_HARQ"; years = list(YEARS); path = PATH_PREDICTIONS
    
    files = []
    if type(years) == str:
        years = [years]
    for year in years:
        if not os.path.exists(path.format(
            model_name = model_name, objective_str = objective_str,
            VARIANT = VARIANT, year = year, extra_path=extra_path)):
            print("File {name} for year {year} is missing".format(
                name=model_name+"_"+objective_str, year=year))
            continue
        files.append(
            pd.read_hdf(path.format(
                model_name = model_name, objective_str = objective_str,
                VARIANT = VARIANT, year = year, extra_path=extra_path),"data"))
        files_columns = files[-1].columns
        files_columns = [column for column in files_columns if column not in ["test","res_naive","res_isds"]]
        files[-1].rename(columns ={
            column:f"{model_name}_{objective_str}_{index}" 
            for index,column in enumerate(files_columns)},inplace=True)
        
        
    files = pd.concat(files)
    return files

class Loss_QLikelihood():
    def __init__(self,**kwargs):
        pass
    def __call__(self, output, target, **kwargs):
        ## treatment of zero values
        if type(target) is pd.Series:
            target = target.to_numpy()
            output = output.to_numpy()
        
        zero = (target==0)
        if zero.sum() == np.prod(target.shape[0]):
            raise ValueError("All entries are zero")
        else:
            smallest_value = target[~zero].min()
            target[zero,...] = smallest_value
            del smallest_value
        
        loss = (target/output-np.log(target/output)-1).mean()
        return loss

def _df_groupby_rolling_cov_worker_(
        identity:int,
        groups:list,
        cov:list,
        groupby:str,
        date_entity:str,
        ):
    while len(groups)>0:
        group = groups.pop()
        column_a,column_b = group.columns
        level_value = group.index.get_level_values(groupby).values[0]
        cov_local = group.droplevel(groupby).rolling(36,12).cov().unstack()[column_a][column_b].\
            to_frame(name="cov")
        cov_local[groupby] = level_value
        cov.append(cov_local)
        
def df_groupby_rolling_cov(
        df:pd.core.frame.DataFrame,
        groupby:str = IDENTIFIER,
        rolling_window:tuple = (36,12),
        date_entity:str = DATE_ENTITY,
        n_workers:int = 32,
        ) -> pd.core.frame.DataFrame:
    '''
    Function that calculates groupby rolling window covariance efficiently

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Dataframe consisting of two columns to take covariances for.
    groupby : str, DAEFAULT IDENTIFIER
        column which to groupby.
    rolling_window : tuple, DEFAULT (36,12)
        tuple consisting of the rolling window parameters.
    date_entity : str, DEFAULT DATE_ENTITY
        entity of the date. 
    n_workers : int, DEFAULT 32
        Number of parallel workers to calculate the task.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    cov : pandas.core.frame.DataFrame 
        Covariance values of the two columns.

    -------------------
    Testing Parameters:
    -------------------
        df = return_data[["test","SDF"]]
        groupby = IDENTIFIER
        rolling_window = (36,12)
        date_entity = DATE_ENTITY
        n_workers = 32
    '''
    df.sort_values(date_entity,inplace=True)
    if len(df.columns)>2:
        raise ValueError("Invalid number of columns. Expected 2, found {:d}, namely ".format(len(df.columns)),
                         df.columns.tolist())
    groups = [y for x,y in df.groupby(groupby)]
    del df
    cov = []    
    
    jobs = []
    for n_worker in range(n_workers):
        # batch = 0
        thread_arguments = {"identity":n_worker,
                            "groups":groups,
                            "cov" : cov,
                            "groupby":groupby,
                            "date_entity":date_entity}
        thread = th.Thread(target=_df_groupby_rolling_cov_worker_,
                           kwargs=thread_arguments)
        jobs.append(thread)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
    
    cov = pd.concat(cov)
    cov=cov.reset_index().set_index([date_entity,groupby],drop=True)
    cov.sort_index(inplace=True)
    return cov

def prediction_performance(
        models_ana,
        model_name:str = "FFNN_w",
        years:list = YEARS,
        datestring = "0724",
        quantiles = 10,
        title_verbose = False,
        beta_option = True,
        R2_lim = (None,None),
        ):
    
    '''
    Calculate prediction performance for MSE objective.
    Calculate beta sorted portfolio performance for SDF.
    
    -------------------
    Testing Parameters:
        models_ana = models +[]
        model_name = "RP"
        
        years           = YEARS
        quantiles       = 10
        datestring      = "0724"
        title_verbose   = False
        beta_option     = False
        
        model_name = "VP"
        model_name = "SDF"
        
        
    '''
    # create empty performance measure set.
    models_ana = copy.deepcopy(models_ana)
    pred_p = {k:
          # {i:
           {j:{} for j in ["R2","R2_modified","QAA","MSE","Q-like","Beta","Beta R2", "Beta EV"]} \
        for k in COLUMNS_RESULTS}
    pred_p["market"] = {i:[] for i in ["Beta","Beta R2", "Beta EV"]}
        
    pred_p_time = {k:
        {j:{} for j in ["R2","R2_modified","QAA","MSE","Q-like","Beta","Beta R2", "Beta EV"]} \
            for k in COLUMNS_RESULTS}
    
    ## constants
    rolling_window=(150,50)
    beta_quantiles = [5,10,20]
    ## import beta factors
    
    sizes = get_series("mcap")
    min_date = "9999-99-99"
        
    qlike = Loss_QLikelihood()
    
    save_model_name = model_name +""
    last_HARQ = False; index = 0
    if model_name == "VP":
        models_ana.append(models_ana[0])
        last_HARQ = True
    
    for model in models_ana:
        # index = len(models_ana)-1; model_name = "RP"; model = models_ana[0]; 
        index +=1
        if last_HARQ and index == len(models_ana):
            model_name = "VP_HARQ"
        name = model["name"]
        print(name,"\n")
        # calculation of scores:
        '''
        First step: Calculate marginal variable influence
        
        per variable, calculate the average values of the output variable given a certain level of input.
            - for sample, for subsamples
        '''
        
        
        return_data = loader_regressions_P3(
                model_name = name,
                years= YEARS,
                objective_str = model_name,
                path = PATH_PREDICTIONS
                )[["test",*COLUMNS_RESULTS]]
        name += model_name
        return_data = pd.merge(return_data,sizes,left_index=True,right_index=True)
        
        ## uncommented if False statement from PhD_PM_2 belongs here
        column = "res_naive";
        
        ## impose restrictions on prediction variables also enforced on output
        if model_name == "RP":
            return_data[column] = return_data[column].clip(-.99,10)
        elif model_name in ["VP","VP_HARQ"]:
            return_data[column] = return_data[column].clip(0,0.1)
        
        ## sizes
        deciles = return_data[column].groupby(DATE_ENTITY).quantile(.9).to_frame(name="dec10")
        deciles["dec1"] = return_data[column].groupby(DATE_ENTITY).quantile(.1)
        return_data = pd.merge(return_data,deciles,left_index=True, right_index=True)
        return_data_dec1 = return_data[return_data[column]<return_data["dec1"]]
        return_data_dec10 = return_data[return_data[column]>return_data["dec10"]]
        
        return_data["year"] = return_data.index.get_level_values("date")
        return_data["year"] = return_data["year"].apply(lambda x: x.year)
        
        
        #### MSE
        pred_p[column]["MSE"][name] = {}
        pred_p[column]["MSE"][name]["Sample"] = ((
            return_data["test"]-return_data[column])**2).mean()
        
        pred_p[column]["MSE"][name]["Small"]  = ((
            return_data_dec1["test"]-return_data_dec1[column])**2).mean()
        pred_p[column]["MSE"][name]["Large"]  = ((
            return_data_dec10["test"]-return_data_dec10[column])**2).mean()
        
        ## by year:
        pred_p_time[column]["MSE"][name] = {}
        return_data = return_data.reset_index().set_index("year")
        pred_p_time[column]["MSE"][name]["sample"] = ((
            return_data["test"]-return_data[column])**2).groupby("year").mean()
        return_data = return_data.reset_index().set_index(["gvkey","date"])
        # pred_p_time[column]["MSE"][name]["Small"]  = ((
        #     return_data_dec1["test"]-return_data_dec1[column])**2).groupby("date").mean().mean()
        # pred_p_time[column]["MSE"][name]["Large"]  = ((
        #     return_data_dec10["test"]-return_data_dec10[column])**2).groupby("date").mean().mean()
        
        
        ## QLikelihood
        if model_name in ["VP","VP_HARQ"]:
            pred_p[column]["Q-like"][name] = {}
            pred_p[column]["Q-like"][name]["Sample"] = qlike(
                return_data[column],return_data["test"])
            pred_p[column]["Q-like"][name]["Small"] = qlike(
                return_data_dec1[column],return_data_dec1["test"])
            pred_p[column]["Q-like"][name]["Large"] = qlike(
                return_data_dec10[column],return_data_dec10["test"],)
            
            
            
        #### R2-pred
        pred_p[column]["R2"][name] = {}
        pred_p[column]["R2"][name]["Sample"] = 1-((
            return_data["test"]-return_data[column])**2).sum()/((
                return_data["test"]-return_data["test"].mean())**2).sum()
        
        pred_p[column]["R2"][name]["Small"]  = 1-((
            return_data_dec1["test"]-return_data_dec1[column])**2).sum()/((
                return_data_dec1["test"]-return_data_dec1["test"].mean())**2).sum()
        pred_p[column]["R2"][name]["Large"]  = 1-((
            return_data_dec10["test"]-return_data_dec10[column])**2).sum()/((
                return_data_dec10["test"]-return_data_dec10["test"].mean())**2).sum()
        
        ## by year:
        
        pred_p_time[column]["R2"][name] = {}
        return_data = return_data.reset_index().set_index("year")
        
        pred_p_time[column]["R2"][name]["sample"] = ((
            return_data["test"]-return_data[column])**2).groupby("year").sum()
        
        pred_p_time[column]["R2"][name]["mean"] = return_data["test"].groupby("year").mean().to_frame("mean")
        return_data = pd.merge(return_data,pred_p_time[column]["R2"][name]["mean"],
                               left_on = "year",right_on = "year", how = "inner")
     
        pred_p_time[column]["R2"][name]["sample"] = 1-(pred_p_time[column]["R2"][name]["sample"]/(
            (return_data["test"]-return_data["mean"])**2).groupby("year").sum())
        
        del pred_p_time[column]["R2"][name]["mean"] 
        return_data = return_data.reset_index().set_index(["gvkey","date"])
            
        
        #### R2-pred
        pred_p[column]["R2_modified"][name] = {}
        pred_p[column]["R2_modified"][name]["Sample"] = 1-((
            return_data["test"]-return_data[column])**2).sum()/(return_data["test"]**2).sum()
        
        pred_p[column]["R2_modified"][name]["Small"]  = 1-((
            return_data_dec1["test"]-return_data_dec1[column])**2).sum()/(return_data_dec1["test"]**2).sum()
        pred_p[column]["R2_modified"][name]["Large"]  = 1-((
            return_data_dec10["test"]-return_data_dec10[column])**2).sum()/(return_data_dec10["test"]**2).sum()
        
        ## by year
        
        
        ## QAA
        pred_p[column]["QAA"][name] = {}
        pred_p[column]["QAA"][name]["Sample"] = QAA(return_data[["test",column]],pred_name = column, date_entity = DATE_ENTITY)
        pred_p[column]["QAA"][name]["Small"]  =  QAA(return_data_dec1[["test",column]],pred_name = column, date_entity = DATE_ENTITY)
        pred_p[column]["QAA"][name]["Large"]  =  QAA(return_data_dec10[["test",column]],pred_name = column, date_entity = DATE_ENTITY)
        
        
        
        
        del return_data_dec1, return_data_dec10
        # normalise weights:
        return_data[column] /= abs(return_data[column]).groupby(DATE_ENTITY).sum()
            
        # beta factor graphs:
            # 1. Sorting by beta factors or predicted returns and graphing beta factor vs excess returns.
        
        if model["name"] not in ["VP","VP_HARQ"] and beta_option:
            # create monthly weighted portfolio WP
            WP = (return_data[column]*return_data["test"]).groupby(DATE_ENTITY).sum().to_frame(name="SDF")
            WP["VAR_SDF"] = WP["SDF"].rolling(*rolling_window).std()**2
            
            return_data = pd.merge(return_data,WP,left_index=True,right_index=True,how="inner")
            del WP
            
            # calculate beta factor
            n_workers = 16
            atime = time.time()
            return_data["cov"] = df_groupby_rolling_cov(
                return_data[["test","SDF"]], date_entity = "date",
                n_workers = n_workers)
            print(":::  df_groupby_rolling_cov  :::    calculation time: {time:f}; \
                  n_workers = {n_wor:d}; n_entities: {n_ent:d}; n_obs: {n_obs:d}".\
                  format(time = round(time.time()-atime,2),
                          n_wor = n_workers, n_ent = len(set(return_data.index.get_level_values(IDENTIFIER))), 
                          n_obs = return_data.shape[0]))
            return_data["beta"] = (return_data["cov"]/return_data["VAR_SDF"]).groupby(IDENTIFIER).shift(1)
            del return_data["cov"]
            
            ## calculate implied explained variation and cross sectional r2
            sum_squared_epsilons = []
            sum_squared_returns = []
            # epsilons_all = []
            return_data.sort_index(level = [DATE_ENTITY,IDENTIFIER])
            data_months = [y for x,y in return_data[["beta","test"]].groupby(DATE_ENTITY)]
            betas = None; returns = None;
            for data_month in data_months:
                # data_month = data_months[0]
                data_month.dropna(inplace=True)
                if data_month.shape[0]<1: continue
                min_date = min(data_month.index.get_level_values(DATE_ENTITY).min(),min_date)
                # epsilons_all.append(data_month.index)
                betas = data_month["beta"].to_numpy()[:,np.newaxis]
                returns = data_month["test"].to_numpy()[:,np.newaxis]
                epsilons = np.matmul((np.identity(betas.shape[0])-np.matmul(
                    np.matmul(betas,np.matmul(betas.T,betas)**-1),betas.T)),returns)
                # epsilons_all[-1] = pd.DataFrame(epsilons,index=epsilons_all[-1],columns=["epsilons"])
                sum_squared_epsilons.append((epsilons**2).mean())
                sum_squared_returns.append((returns**2).mean())
            
            # epsilons_all = pd.concat(epsilons_all)
            # return_data["epsilon"] = epsilons_all
            pred_p[column]["Beta EV"][name] = 1-np.mean(sum_squared_epsilons)/np.mean(sum_squared_returns)
            del data_months, sum_squared_epsilons, sum_squared_returns, betas, returns
            
            # return_data["impl ret"] =  return_data["beta"]*return_data["SDF"]
            # pred_p[column]["Beta XS-R2"] = 1-(
            #     (return_data["epsilon"].groupby(IDENTIFIER).sum()**2/return_data["epsilon"].groupby(IDENTIFIER).count()).mean()/\
            #     (return_data["impl ret"].groupby(IDENTIFIER).sum()**2/return_data["impl ret"].groupby(IDENTIFIER).count()).mean())
            
            ## quantiles of beta factors 
            shape = return_data.groupby(DATE_ENTITY).size().to_frame("shape")
            return_data = pd.merge(return_data, shape, left_index=True,right_index=True,how="inner")
            return_data["beta_rank"] = return_data["beta"].groupby(DATE_ENTITY).rank(method="average")/\
                return_data["shape"]
            
            pred_p[column]["Beta"][name] = {}; pred_p[column]["Beta R2"][name] = {}; 
            for quantiles in beta_quantiles:
                return_data[f"Beta_{quantiles:d}q"]  = return_data["beta_rank"].transform(lambda x:pd.qcut(
                    x,quantiles,labels=False, duplicates="drop"))
                q_returns =  return_data.groupby(f"Beta_{quantiles:d}q")[["beta","test"]].mean()
                reg = sc.stats.linregress(q_returns["beta"],q_returns["test"])
                q_returns["reg"] = q_returns["beta"]*reg.slope+reg.intercept
                pred_p[column]["Beta"][name][f"{quantiles:d}"] = q_returns.copy()
                pred_p[column]["Beta R2"][name][f"{quantiles:d}"] = reg.rvalue**2
                del q_returns
            
            return_data = return_data[["test",*COLUMNS_RESULTS,"mcap"]]
                
    ## market return beta:
    # min_date = "2004-01-31"
    if False:
        min_date = int(min_date[:4])*10000+int(min_date[5:7])*100+int(min_date[8:])
        returns_data_gu = pd.merge(pd.read_hdf(PATH_DATA+"gu_ret.h5","data"),
                                   pd.read_hdf(PATH_DATA+"beta.h5","data"),
                                   left_index=True,right_index=True)
        returns_data_gu = returns_data_gu[returns_data_gu.index.get_level_values(DATE_ENTITY)>=min_date]
        shape = returns_data_gu.groupby(DATE_ENTITY).size().to_frame("shape")
        returns_data_gu = pd.merge(returns_data_gu, shape, left_index=True,right_index=True,how="inner")
        del shape
        returns_data_gu["beta_rank"] = returns_data_gu["B"].groupby(DATE_ENTITY).rank(method="average")/\
            returns_data_gu["shape"]
        FamaMacBethBeta = {}
        for quantile in beta_quantiles:
            returns_data_gu[f"Beta_{quantile:d}q"]  = returns_data_gu["beta_rank"].transform(lambda x:pd.qcut(
                x,quantile,labels=False, duplicates="drop"))
            q_returns =  returns_data_gu.groupby(f"Beta_{quantile:d}q")[["B","ret"]].mean()
            reg = sc.stats.linregress(q_returns["B"],q_returns["ret"])
            q_returns["reg"] = q_returns["B"]*reg.slope+reg.intercept
            FamaMacBethBeta[f"{quantile:d}"] = {"data":q_returns.copy(),"R2":reg.rvalue**2}
            del q_returns
        
        for column_result in COLUMNS_RESULTS:
            print("\n"+column_result,"Beta R2 table:\n")
            print("","market",*pred_p[column_result]["Beta R2"].keys(),sep=" & ")
            for quantile in beta_quantiles:
                print(quantile,"{:5.2f}".format(FamaMacBethBeta[f"{quantile:d}"]["R2"]*100),sep=" & ", end=" & ")
                for model_name in pred_p[column_result]["Beta R2"].keys():
                    print("{:5.2f}".format(pred_p[column_result]["Beta R2"][model_name]\
                                                   [f"{quantile:d}"]*100),end=" & ")
                print()
    ##############
    #  Plotting  #
    ##############
    
    # linewidth = 2.5
    '''
    R2:
        sum up all R2 values into dataframe.
        plot.
    
    '''
    for R2_name in ["R2","R2_modified"]:
        for column in COLUMNS_RESULTS:
            column_title = column.split("_")[1].capitalize()
            
            ### R2
            fontsize = 12
            
            R2_data = pd.DataFrame(pred_p[column][R2_name]).T
            R2plot=(R2_data*100).T.plot.bar(rot=0,edgecolor="white", linewidth = 1,width = .85,
                            fontsize = fontsize)#, width=space_per_bar)
            
            R2_lim_loc = [-.5,.05]
            if type(R2_lim) == float:
                R2_lim_loc[0] = R2_lim+0
            else:
                for index_i in range(2):
                    if R2_lim[index_i] is not None:
                        R2_lim_loc[index_i] = R2_lim[index_i]+0
                
            R2plot.set_ylim(max(min(-0.05,*R2_data.min()*105),R2_lim_loc[0]),max(*R2_data.max()*105,R2_lim_loc[1]))
            R2plot.grid(True,which="both")
            # # plt.legend(prop={'size': fontsize-0.5},
            # #            bbox_to_anchor=(-0.025, 1.03, 1.1, .102), loc='lower left',
            # #            borderaxespad=0.,ncol = n_col)
            R2plot.tick_params(axis="x",labelrotation = 15)
            shift_down = -.25
            if len(pred_p[COLUMNS_RESULTS[0]]["R2"])>4:
                shift_down = -.35    
            R2plot.legend(
                prop={'size': fontsize-0.5},
                bbox_to_anchor=(-.12, shift_down, 0.25, 1), loc='lower left',
                borderaxespad=0.,ncol =4)
            if title_verbose: plt.title(column_title+" R2")
            R2plot.get_figure().savefig(PATH_PROCESSED.format(
                date=datestring,model_name=model_name,VARIANT = VARIANT,name_type=R2_name,fileending="png"),
            dpi=200,bbox_inches = "tight")
    
    for column in COLUMNS_RESULTS:
        ### QAA
        '''
        Reshape the data from QAA to fit the graph.
        plot.
        '''
        fontsize = 18
        
        ## reshape.
        QAA_data = {}
        QAA_index = -1
        y_lim_min = -.05; y_lim_max = .05;
        for QAA_name in ["QAA","QAD", "HML QAA", "HML QAD"]:
            ## reshaping
            QAA_data[QAA_name] = {}
            QAA_index+=1
            for model_name in pred_p[column]["QAA"]:
                QAA_data[QAA_name][model_name] = {}
                for sample_name in ["Sample","Small","Large"]:
                    QAA_data[QAA_name][model_name][sample_name] = pred_p[column]["QAA"][model_name][sample_name][QAA_index]
            QAA_data[QAA_name] = pd.DataFrame(QAA_data[QAA_name])
            y_lim_min = min(y_lim_min,QAA_data[QAA_name].min().min())
            y_lim_max = max(y_lim_max,QAA_data[QAA_name].max().max())
            
        ## plot
         
        QAA_index = -1
        grouped_fig, grouped_axes = plt.subplots(
            2, 2, figsize=(6*2, 6*2))
        if title_verbose: grouped_fig.suptitle(column_title, fontsize=fontsize)
        for QAA_name in ["QAA","QAD", "HML QAA", "HML QAD"]:
            
            QAA_index+=1
            index = (QAA_index//2,QAA_index%2)
            QAA_data[QAA_name].plot.bar(ax=grouped_axes[index],fontsize=fontsize,
                rot=0, edgecolor="white", linewidth = 3, width = .9,legend=False)
            # index = list(index); index[0]+=1; index=tuple(index);
            # grouped_axes[index].set_xlabel("Time",fontsize=fontsize)
            grouped_axes[index].set_ylabel("Value",fontsize=fontsize)
            grouped_axes[index].set_ylim(y_lim_min,y_lim_max)
            
            grouped_axes[index].set_title(QAA_name,fontsize=fontsize,loc="center")
            grouped_axes[index].grid()
        handles, labels = grouped_axes[(1,0)].get_legend_handles_labels()
        grouped_fig.legend(
            handles, labels,
            prop={'size': fontsize-0.5},
            # fancybox=True, shadow=True, 
            # bbox_to_anchor=(0., 1.02, 1., .5*HEIGHT_LEGEND+.002), loc='lower left',
            bbox_to_anchor=(0, -0.045, 0.25, 1), loc='lower left',
            borderaxespad=0.,ncol = 6)
        if title_verbose: grouped_fig.set_title(column_title+ " QAA",fontsize=fontsize,loc="center")
        grouped_fig.tight_layout()
        grouped_fig.show()
        grouped_fig.savefig(PATH_PROCESSED.format(
            date=datestring,model_name=save_model_name,VARIANT = VARIANT,name_type="QAA",fileending="png"),
            dpi=200,bbox_inches = "tight")
        
        ## Betas
        if False:
            plt.rcParams.update({'font.size': fontsize})
            grouped_fig, grouped_axes = plt.subplots(
                1, 3, figsize=(6*3, 6*1))
            if title_verbose: grouped_fig.suptitle(column_title, fontsize=fontsize)
            index_beta_plot = 0
            matplotlib_default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for quantile in beta_quantiles:
                grouped_axes[index_beta_plot].plot(
                    FamaMacBethBeta[str(quantile)]["data"]["B"].to_numpy(), 
                    FamaMacBethBeta[str(quantile)]["data"]["ret"].to_numpy(),
                    marker = "o", markerfacecolor='None', linestyle='None',
                    color = "black", linewidth = 2)
                grouped_axes[index_beta_plot].plot(
                    FamaMacBethBeta[str(quantile)]["data"]["B"].to_numpy(), 
                    FamaMacBethBeta[str(quantile)]["data"]["reg"].to_numpy(), 
                    color = "black",label ="market beta", linewidth = 2
                    )
                color_index=0
                for model_name in pred_p[column]["Beta"].keys():
                    grouped_axes[index_beta_plot].plot(
                        pred_p[column]["Beta"][model_name][str(quantile)]["beta"].to_numpy(),
                        pred_p[column]["Beta"][model_name][str(quantile)]["test"].to_numpy(),
                        marker = "o", markerfacecolor='None', linestyle='None',
                        color = matplotlib_default_colors[color_index], linewidth = 2)
                    
                    grouped_axes[index_beta_plot].plot(
                        pred_p[column]["Beta"][model_name][str(quantile)]["beta"].to_numpy(), 
                        pred_p[column]["Beta"][model_name][str(quantile)]["reg"].to_numpy(), 
                        color = matplotlib_default_colors[color_index], 
                        label=f"{model_name:s} beta", linewidth = 2)
                    color_index+=1
                if index_beta_plot == 0:
                    grouped_axes[index_beta_plot].set_ylabel("Avg Returns",fontsize=fontsize)
                grouped_axes[index_beta_plot].set_xlabel("Beta",fontsize=fontsize)
                grouped_axes[index_beta_plot].grid()
                
                index_beta_plot += 1
            handles, labels = grouped_axes[0].get_legend_handles_labels()
            handles.extend([Line2D([0], [0], color="lightgrey",lw=2, marker='o'),
                            Line2D([0],[0],color = "lightgrey",lw=2)])
            labels.extend(["Quantiles", "Fitted Line"])
            
            shift_down = -0.045
            if len(pred_p[COLUMNS_RESULTS[0]]["R2"])>4:
                shift_down = -0.085
            grouped_fig.legend(
                handles, labels,
                prop={'size': fontsize-0.5},
                bbox_to_anchor=(0, -0.045, 0.25, 1), loc='lower left',
                borderaxespad=0.,ncol =7)
            grouped_fig.tight_layout()
            grouped_fig.show()
            grouped_fig.savefig(PATH_PROCESSED.format(
                date=datestring,model_name=save_model_name,VARIANT = VARIANT,name_type="Beta",fileending="png"),
                dpi=200,bbox_inches = "tight")
            ## Other
            '''
            What do we plot here?
            We have left Beta, Beta R2, Beta EV
            '''
        
    return pred_p, pred_p_time


def portfolio_performance(
        models_ana,
        model_name:str = "FFNN_w",
        years:list = YEARS,
        datestring = "0724",
        quantiles = 10,
        ):
    '''
    This function shows the monthly performance for the portfolios. 
    This includes performance graphs of both methods for weighted portfolios and HML portfolios.
    Additionally, for all portfolios, performance measures are calculated.
    
    Returns:
    --------
        pp: dict
            Results organised by forecast combination method, result type, portfolio type
    -------------------
    Testing Parameters:
        model_name      = "RP"
        
        models_ana      = models[:3]
        years           = YEARS
        datestring      = "1024" 
        quantiles       = 10
        
        model_name = "SDF"
        
    '''
    # name_stage = model_name
    ann_adj = 52**.5
    
    HIGH_QUANTILE = str(quantiles-1)
    market_columns = ["market_ev","market_vv"]
    
    pp = {k:
          # {i:
           {j:{} for j in ["total portfolio","monthly performance","monthly portfolio"]} \
            # for i in ["managed","HML equal","HML value"]} 
        for k in COLUMNS_RESULTS}
    
    value_weights = np.exp(pd.read_hdf(PATH_DATA+"mcap.h5","data"))
    
    for model in models_ana:
        # model = models_ana[0]
        
        name = model["name"]
        print_hint2("Calculating Performance for "+name)
        
        return_data = loader_regressions_P3(
                model_name = name,
                years= YEARS,
                objective_str = model_name,
                path = PATH_PREDICTIONS
                )[["test",*COLUMNS_RESULTS]]
        
        return_data = pd.merge(
            value_weights, return_data, left_index=True,
            right_index=True,how = "inner"
            )
        
        # monthly_portfolio[name] = pd.DataFrame() #del
        # monthly_performance[name] = pd.DataFrame() #del
        
        for COLUMN in COLUMNS_RESULTS:
            # COLUMN = COLUMNS_RESULTS[0]
            # subname = name + "_"+COLUMN.split("_")[1]
            
            
            deciles = pd.DataFrame(columns = ["1",HIGH_QUANTILE])
            
            deciles["1"]=return_data[COLUMN].groupby(DATE_ENTITY).quantile(q=1/quantiles)
            
            deciles[HIGH_QUANTILE]=return_data[COLUMN].groupby(DATE_ENTITY).quantile(q=1-1/quantiles)
            
            return_data_local =\
                pd.merge(return_data[["test",COLUMN,"mcap"]],deciles, how="inner", 
                         left_index = True, right_index=True)
                         # left_on=DATE_ENTITY,right_on=DATE_ENTITY)
                         
            # HML portfolios ev and vv, managed portfolio
            low_data = return_data_local.loc[
                return_data_local[COLUMN]<return_data_local["1"],["test","mcap"]]
            high_data = return_data_local.loc[
                return_data_local[COLUMN]>return_data_local[HIGH_QUANTILE],["test","mcap"]]
            
            # turnover EV
            return_data_local["ev_long"] = 0; return_data_local["ev_short"] = 0;
            return_data_local.loc[return_data[COLUMN]>return_data_local[HIGH_QUANTILE],"ev_long"] = 1
            return_data_local.loc[return_data[COLUMN]<return_data_local["1"],"ev_short"] = 1
            return_data_local["ev_long_lag"] = (return_data_local["ev_long"]*(1+return_data_local["test"])).groupby(IDENTIFIER).shift(1).fillna(0)
            return_data_local["ev_short_lag"] = (return_data_local["ev_short"]*(1+return_data_local["test"])).groupby(IDENTIFIER).shift(1).fillna(0)
            return_data_local["ev_long_lag"] = return_data_local["ev_long_lag"]/return_data_local["ev_long_lag"].groupby(DATE_ENTITY).sum()
            return_data_local["ev_short_lag"] = return_data_local["ev_short_lag"]/return_data_local["ev_short_lag"].groupby(DATE_ENTITY).sum()
            return_data_local["ev_long"] = return_data_local["ev_long"]/return_data_local["ev_long"].groupby(DATE_ENTITY).sum()
            return_data_local["ev_short"] = return_data_local["ev_short"]/return_data_local["ev_short"].groupby(DATE_ENTITY).sum()
            # turnover VV
            return_data_local["vv_long"] = 0; return_data_local["vv_short"] = 0;
            return_data_local.loc[return_data[COLUMN]>return_data_local["9"],"vv_long"] = 1
            return_data_local.loc[return_data[COLUMN]<return_data_local["1"],"vv_short"] = 1
            return_data_local["vv_long"]*=return_data_local["mcap"]
            return_data_local["vv_short"]*=return_data_local["mcap"]
            return_data_local["vv_long"] = return_data_local["vv_long"]/return_data_local["vv_long"].groupby(DATE_ENTITY).sum()
            return_data_local["vv_short"] = return_data_local["vv_short"]/return_data_local["vv_short"].groupby(DATE_ENTITY).sum()
            return_data_local["vv_long_lag"] = return_data_local["vv_long"].groupby(IDENTIFIER).shift(1).fillna(0)
            return_data_local["vv_short_lag"] = return_data_local["vv_short"].groupby(IDENTIFIER).shift(1).fillna(0)
                
            
            
            
            pp[COLUMN]["monthly portfolio"][name] = pd.DataFrame()
            pp[COLUMN]["monthly portfolio"][name]["low_ev"] = low_data["test"].groupby(DATE_ENTITY).\
                    mean()
            pp[COLUMN]["monthly portfolio"][name]["high_ev"] = high_data["test"].groupby(DATE_ENTITY).\
                    mean()
            pp[COLUMN]["monthly portfolio"][name]["HML_ev"] = \
                pp[COLUMN]["monthly portfolio"][name]["high_ev"]-\
                    pp[COLUMN]["monthly portfolio"][name]["low_ev"]
                    
            pp[COLUMN]["monthly portfolio"][name]["low_vv"] = (low_data["test"]*low_data["mcap"]).\
                groupby(DATE_ENTITY).sum()/low_data["mcap"].groupby(DATE_ENTITY).sum()
            pp[COLUMN]["monthly portfolio"][name]["high_vv"] = (high_data["test"]*high_data["mcap"]).\
                groupby(DATE_ENTITY).sum()/high_data["mcap"].groupby(DATE_ENTITY).sum()
            
            pp[COLUMN]["monthly portfolio"][name]["HML_vv"] = \
                pp[COLUMN]["monthly portfolio"][name]["high_vv"]-\
                    pp[COLUMN]["monthly portfolio"][name]["low_vv"]
              
            # managed portfolio
            return_data[COLUMN] /= abs(return_data[COLUMN]).groupby(DATE_ENTITY).sum()
            pp[COLUMN]["monthly portfolio"][name]["managed"] = (return_data["test"]*return_data[COLUMN]).\
                groupby(DATE_ENTITY).sum()
            # # restricted managed
            # return_data[COLUMN] -= return_data[COLUMN].groupby(DATE_ENTITY).min()
            # return_data[COLUMN] /= abs(return_data[COLUMN]).groupby(DATE_ENTITY).sum()
            # pp[COLUMN]["monthly portfolio"][name]["managed_r"] = (return_data["test"]*return_data[COLUMN]).\
            #     groupby(DATE_ENTITY).sum()
            
            # # ranked managed
            # return_data["ranks"] = return_data[COLUMN].groupby(DATE_ENTITY).rank(ascending=True)/\
            #     return_data[COLUMN].groupby(DATE_ENTITY).count()
            # return_data["ranks"] -= .25
            # return_data["ranks"] /= return_data["ranks"].abs().groupby(DATE_ENTITY).sum()
            # pp[COLUMN]["monthly portfolio"][name]["managed_r"] = (return_data["test"]*return_data["ranks"]).\
            #     groupby(DATE_ENTITY).sum()
                
            if "market" not in pp.keys():
                pp[COLUMN]["monthly portfolio"][name]["market_ev"] = \
                    return_data["test"].groupby(DATE_ENTITY).mean()
                pp[COLUMN]["monthly portfolio"][name]["market_vv"] = \
                    (return_data["test"]*return_data["mcap"]).groupby(DATE_ENTITY).sum()/\
                        return_data["mcap"].groupby(DATE_ENTITY).sum()
                        
            '''
            Performance stats:
                mean,   std,    SR,     SR annual,  sortino ratio,  maximum drawdown
                max gain, min gain, VaR 99.5, 
                beta to market,     alpha to market,    alpha to factor models,
                
            '''
            pp[COLUMN]["monthly performance"][name] = pd.DataFrame()
            columns_portfolios = list(pp[COLUMN]["monthly portfolio"][name].columns)
            pp[COLUMN]["monthly portfolio"][name].loc[:,"year"] =  pp[COLUMN]["monthly portfolio"][name].\
                index.get_level_values(DATE_ENTITY)
            pp[COLUMN]["monthly portfolio"][name].loc[:,"year"] = \
                pp[COLUMN]["monthly portfolio"][name]["year"].apply(lambda x: x.year)
            # mean, std
            pp[COLUMN]["monthly performance"][name]["Mean"] =pp[COLUMN]["monthly portfolio"][name][columns_portfolios].mean()
            pp[COLUMN]["monthly performance"][name]["Std"] = pp[COLUMN]["monthly portfolio"][name][columns_portfolios].std()
            # sharpe ratio
            pp[COLUMN]["monthly performance"][name]["SR"] = pp[COLUMN]["monthly performance"][name]["Mean"]/\
                pp[COLUMN]["monthly performance"][name]["Std"]
            pp[COLUMN]["monthly performance"][name]["SR annualized"] = pp[COLUMN]["monthly performance"][name]["SR"]*ann_adj
            # annual values
            pp[COLUMN]["monthly performance"][name]["Mean annual"] = ((pp[COLUMN]["monthly portfolio"][name].\
                                                                       set_index("year")+1).groupby("year").prod()-1).mean()
            pp[COLUMN]["monthly performance"][name]["Std annual"] = ((pp[COLUMN]["monthly portfolio"][name].\
                                                                      set_index("year")+1).groupby("year").prod()-1).std()
            pp[COLUMN]["monthly performance"][name]["SR annual"] = pp[COLUMN]["monthly performance"][name]["Mean annual"]/\
                pp[COLUMN]["monthly performance"][name]["Std annual"]
            
            # return_data_local = pd.merge(return_data_local,
            #                              (pp[COLUMN]["monthly portfolio"][name][columns_portfolios]+1).shift(1).fillna(1),
            #                              left_index = True, right_index=True, how = "inner")
            
            return_data_local["mcap"] /= return_data_local["mcap"].groupby(DATE_ENTITY).sum()
            pp[COLUMN]["monthly performance"][name]["turnover"] = 0.0
            pp[COLUMN]["monthly performance"][name]["adj turnover"] = 0.0
            for _weighting_ in ["vv","ev"]:
                for _half_,_source_ in [(f"{_weighting_}_short","low"),(f"{_weighting_}_long","high")]:
                    # _weighting_ = "vv"; _half_ = f"{_weighting_}_long"; _source_ = "high"
                    series = return_data_local[_half_]-return_data_local[_half_].groupby(IDENTIFIER).shift(1).fillna(0)
                    pp[COLUMN]["monthly performance"][name].loc[f"{_source_}_{_weighting_}","turnover"] = \
                        series[series>0].groupby(DATE_ENTITY).sum().mean()
                    series = return_data_local[f"{_half_}_lag"]-return_data_local[_half_]
                    pp[COLUMN]["monthly performance"][name].loc[f"{_source_}_{_weighting_}","adj turnover"] = \
                        series[series>0].groupby(DATE_ENTITY).sum().mean()
                    del return_data_local[_half_], return_data_local[f"{_half_}_lag"]
                # HML
                pp[COLUMN]["monthly performance"][name].loc[f"HML_{_weighting_}","turnover"] = \
                    pp[COLUMN]["monthly performance"][name].loc[f"low_{_weighting_}","turnover"]+\
                        pp[COLUMN]["monthly performance"][name].loc[f"high_{_weighting_}","turnover"]
                pp[COLUMN]["monthly performance"][name].loc[f"HML_{_weighting_}","adj turnover"] = \
                    pp[COLUMN]["monthly performance"][name].loc[f"low_{_weighting_}","adj turnover"]+\
                        pp[COLUMN]["monthly performance"][name].loc[f"high_{_weighting_}","adj turnover"]+\
                            pp[COLUMN]["monthly performance"][name].loc[f"HML_{_weighting_}","Mean"]/2
            if "market" not in pp.keys():
                series = return_data_local["mcap"]-return_data_local["mcap"].groupby(IDENTIFIER).shift(1).fillna(0)
                series_group = abs(series).groupby(DATE_ENTITY).sum()
                series_group = series_group.iloc[1:]
                pp[COLUMN]["monthly performance"][name].loc["market_vv",["turnover","adj turnover"]] = \
                    series_group.mean()/2
                series = (return_data_local["test"]+1).shift(1).fillna(0)
                series /= series.groupby(DATE_ENTITY).sum()
                series2 = pd.Series(1,index=series.index)
                series2/=series2.groupby(DATE_ENTITY).sum()
                series3 = series-series2
                pp[COLUMN]["monthly performance"][name].loc["market_ev",["turnover","adj turnover"]] = \
                    abs(series3).groupby(DATE_ENTITY).sum().mean()/2
            
            return_data_local[COLUMN]/=return_data_local[COLUMN].abs().groupby(DATE_ENTITY).sum()
            series = return_data_local[COLUMN] - return_data_local[COLUMN].groupby(IDENTIFIER).shift(1).fillna(0)
            pp[COLUMN]["monthly performance"][name].loc["managed",["turnover","adj turnover"]] = \
                series[series>0].groupby(DATE_ENTITY).sum().mean()
            # pp[COLUMN]["monthly performance"][name].loc["managed_r",["turnover","adj turnover"]] = \
            #     series[series>0].groupby(DATE_ENTITY).sum().mean()
            del series
            
            pp[COLUMN]["monthly performance"][name].loc[:,"cosine similarity"] =\
                ((return_data_local["mcap"]*return_data_local[COLUMN]).groupby(DATE_ENTITY).sum()/(
                    (return_data_local["mcap"]**2).groupby(DATE_ENTITY).sum()**.5*\
                        (return_data_local[COLUMN]**2).groupby(DATE_ENTITY).sum()**.5)).mean()

            #sortino ratio
            pp[COLUMN]["monthly performance"][name]["Std Sortino"] = 0.0
            threshold_sortino = 0
            for columnname in columns_portfolios:
                pp[COLUMN]["monthly performance"][name].loc[columnname,"Std Sortino"] = \
                    pp[COLUMN]["monthly portfolio"][name].loc[
                        pp[COLUMN]["monthly portfolio"][name][columnname]<threshold_sortino,columnname].std()
            pp[COLUMN]["monthly performance"][name]["Sortino ratio"] = pp[COLUMN]["monthly performance"][name]["Mean"]/\
                pp[COLUMN]["monthly performance"][name]["Std Sortino"]
            pp[COLUMN]["monthly performance"][name]["Sortino ratio annualized"] = \
                pp[COLUMN]["monthly performance"][name]["Sortino ratio"]*ann_adj
            
            # min, max, VaR return
            pp[COLUMN]["monthly performance"][name]["Min"] =pp[COLUMN]["monthly portfolio"][name][columns_portfolios].min()
            pp[COLUMN]["monthly performance"][name]["Max"] =pp[COLUMN]["monthly portfolio"][name][columns_portfolios].max()
            pp[COLUMN]["monthly performance"][name]["Var 99.5%"] =pp[COLUMN]["monthly portfolio"][name][columns_portfolios].\
                quantile(.005)
            pp[COLUMN]["monthly performance"][name]["Var 95%"] =pp[COLUMN]["monthly portfolio"][name][columns_portfolios].\
                quantile(.05)
            
            # ratio of positive returns
            pp[COLUMN]["monthly performance"][name]["Pos ratio"] = 0.0
            for columnname in columns_portfolios:
                pp[COLUMN]["monthly performance"][name].loc[columnname,"Pos ratio"] = \
                    pp[COLUMN]["monthly portfolio"][name][pp[COLUMN]["monthly portfolio"][name][columnname]>=0][columnname].count()/\
                    pp[COLUMN]["monthly portfolio"][name].shape[0]
            
            # maximum drawdown
            pp[COLUMN]["total portfolio"][name] = np.log(pp[COLUMN]["monthly portfolio"][name][columns_portfolios]+1).cumsum()
            pp[COLUMN]["monthly performance"][name]["Drawdown max"] = 1-np.exp(-
                (pp[COLUMN]["total portfolio"][name][columns_portfolios].cummax()-pp[COLUMN]["total portfolio"][name][columns_portfolios]).max())
            
            if "market" not in pp.keys():
                pp["market"] = {}
                pp["market"]["total portfolio"] = pp[COLUMN]["total portfolio"][name][market_columns]
                pp["market"]["monthly portfolio"] = pp[COLUMN]["monthly portfolio"][name][market_columns]
                pp["market"]["monthly performance"] = pp[COLUMN]["monthly performance"][name].loc[market_columns,:]
                pp[COLUMN]["total portfolio"][name].drop(market_columns,axis=1,inplace=True)
                pp[COLUMN]["monthly portfolio"][name].drop(market_columns,axis=1,inplace=True)
                pp[COLUMN]["monthly performance"][name].drop(market_columns,axis=0,inplace=True)
                
                
                
            # beta, alpha, factor models.
            pp[COLUMN]["monthly performance"][name]["beta"] = 0 
            pp[COLUMN]["monthly performance"][name]["alpha"] = 0 
            pp[COLUMN]["monthly performance"][name]["alpha t"] = 0 
            for portfolio_column in COLUMNS_PORTFOLIOS_DEFAULT:
                # portfolio_column = "HML_vv"
                market_comp = "market_ev"
                if portfolio_column in ["managed","HML_vv","high_vv","low_vv"]:
                    market_comp = "market_vv"
                cov = np.cov(pp[COLUMN]["monthly portfolio"][name][portfolio_column],
                             pp["market"]["monthly portfolio"][market_comp])
                
                pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"beta"] = \
                    cov[1,0]/cov[0,0]
                pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"alpha"] = \
                    pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"Mean"]-\
                    pp["market"]["monthly performance"].loc[market_comp,"Mean"]*\
                        pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"beta"]
                
                X = smapi.add_constant(pp["market"]["monthly portfolio"][[market_comp]])
                model = smapi.OLS(pp[COLUMN]["monthly portfolio"][name][[portfolio_column]],
                                  X).fit()
                pp[COLUMN]["monthly performance"][name].loc[portfolio_column, "alpha t"] = model.tvalues[0]
               
                
            # !!! here we have to rebuild FF5 portfolios
            if False:
                FF5_portfolios = pd.read_hdf(PATH_FF5_file,"data")
                FF5_portfolios.reset_index(inplace=True)
                FF5_portfolios["month"] = FF5_portfolios["month"].apply(
                    lambda x: str(x)[:4]+"-"+str(x)[4:6]+"-"+str(x)[6:])
                FF5_portfolios.set_index("month",inplace=True,drop=True)
                FF5_portfolios = FF5_portfolios["2003-01-31":]
                
                
                pp[COLUMN]["monthly performance"][name]["alpha_FF3"] = 0
                pp[COLUMN]["monthly performance"][name]["alpha_FF3_t"] = 0
                pp[COLUMN]["monthly performance"][name]["alpha_FF5+M"] = 0
                pp[COLUMN]["monthly performance"][name]["alpha_FF5+M_t"] = 0
                
                for portfolio_column in COLUMNS_PORTFOLIOS_DEFAULT:
    
                    X = smapi.add_constant(FF5_portfolios[["market","mcap","BM"]])
                    model = smapi.OLS(pp[COLUMN]["total portfolio"][name][portfolio_column].to_frame(),
                                      X).fit()   
                    pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"alpha_FF3"] = model.params["const"]
                    pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"alpha_FF3_t"] = model.tvalues["const"]
                    
                    X = smapi.add_constant(FF5_portfolios[["market","mcap","BM", 'CIN', 'OP', 'M6M']])
                    model = smapi.OLS(pp[COLUMN]["total portfolio"][name][portfolio_column].to_frame(),
                                      X).fit()   
                    pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"alpha_FF5+M"] = model.params["const"]
                    pp[COLUMN]["monthly performance"][name].loc[portfolio_column,"alpha_FF5+M_t"] = model.tvalues["const"]
            
    # columns_portfolio = [column for column in monthly_portfolio[name].columns if column != "year"]
    # total_portfolio[name] = (monthly_portfolio[name][columns_portfolio]+1).prod(axis=0)-1
    ##################################################
    ##################################################
    ##########  Saving tables and plotting  ##########
    ##################################################
    ##################################################
    
    '''
    Data has multiple branches:
        fc-[total portfolio, monthly performance, monthly portfolio]
        2x3 six tables.
        1 overarching name.
        make folder with overarching name.
            - save excel tables of the six tables for each model
    '''
    
    PORTFOLIO_PERIOD = pp["market"]["total portfolio"]["market_vv"].shape[0]
    # graphs = {}
    
    n_columns = 2 # number of different loss functions
    # n_graphs = 2 #number of different forecast combination schemes
    n_rows = int((len(pp[COLUMNS_RESULTS[0]]["total portfolio"])-1)/(n_columns))+1 #number of different model setups
    
    # figures = {
    #     "res_isds": plt.subplots(n_rows,n_columns,figsize=(28,12)),
    #     "res_naive":plt.subplots(n_rows,n_columns,figsize=(28,12))}
    # indices = {
    #     "res_isds": 0,
    #     "res_naive":0}
    
    color = "black"
    linewidth = 2.5
    fontsize = 21
    for COLUMN in COLUMNS_RESULTS:
        column_graph_limits = [0,0]
        #########################
        ########  Saving  #######
        #########################
        for TABLE in ['total portfolio', 'monthly performance', 'monthly portfolio']:
            for MODEL in pp[COLUMN][TABLE].keys():
                # COLUMN = "res_isds"; TABLE = "total portfolio";MODEL="7x3_rp"
                pp[COLUMN][TABLE][MODEL].to_csv(
                    PATH_PROCESSED.format(
                        date=datestring,model_name=model_name,VARIANT = VARIANT,name_type=TABLE,fileending="csv"))
                    
       
                #########################
                #######  Plotting  ######
                #########################
                ### create graphs and save them.
                if TABLE != "total portfolio":
                    continue
                # COLUMN = "res_isds"; TABLE = "total portfolio";MODEL="7x3_rp"
                
                title = model_name+": "+MODEL#+":"+COLUMN.split("_")[1]
                # title = title.replace("_"," ")
                fig,mpg=plt.subplots(figsize=(14,6))
                
                # graphs[COLUMN+" "+MODEL] = (fig,mpg)
                # pp[COLUMN][TABLE][MODEL].plot()
                
                # mpg.set_title("High-Minus-Low Portfolio Returns vs Benchmarks")
                
                mpg.set_xlabel("Time",fontsize=fontsize)
                mpg.set_ylabel("Log return",color=color,fontsize=fontsize)
                mpg.plot(pp["market"]["total portfolio"]["market_ev"],
                         color="black", label="market ev",lw=linewidth+1)
                mpg.plot(pp["market"]["total portfolio"]["market_vv"],
                         color="grey", label="market vv",lw=linewidth+1)
                
                for PORTFOLIO in COLUMNS_PORTFOLIOS_DEFAULT:
                    mpg.plot(pp[COLUMN][TABLE][MODEL][PORTFOLIO],
                             label=PORTFOLIO.replace("_"," "),lw=linewidth)
                # xtic=[*range(0,PORTFOLIO_PERIOD,48),PORTFOLIO_PERIOD-PORTFOLIO_PERIOD%12]
                # del xtic[-2]
                # if xtic[-1] == xtic[-2]: xtic = xtic[:-1]
                # HEIGHT_LEGEND = (len(COLUMNS_PORTFOLIOS_DEFAULT)+2)//5+1
                
                # legend_lines = [*[Line2D([0],[0],color = i[3],lw=2) for i in input_models],Line2D([0],[0],color = "lightgrey",lw=3),
                #                 Line2D([0],[0],color = "black",lw=2,ls="solid"),
                #                 Line2D([0],[0],color = "black",lw=2,ls="dotted"),
                #                 Line2D([0],[0],color = "black",lw=2,ls=(5,(10,3)) ),
                #                 Line2D([0],[0],color = "black",lw=2,ls=(0,(3,3,1,3))),
                #                 # Line2D([0],[0],color = "black",lw=2,ls="dashdot")
                #                 ]
                mpg.legend(
                    # legend_lines,[*[i[2] for i in input_models],"S&P500","Long ISDS","Short ISDS",
                    #                      "Long naive", "Short Naive"
                    #                      # "HML"
                    #                      ],
                    #bbox_to_anchor=(1.03,0.8),
                    prop={'size': fontsize-0.5},
                    # bbox_to_anchor=(0., 1.02, 1., .5*HEIGHT_LEGEND+.002), loc='lower left',
                    bbox_to_anchor=(1.02, 0.00, 0.25, 1), loc='lower left',
                    borderaxespad=0.,ncol = 1)
                
                # mpg.set_xticks(xtic)
                
                column_graph_limits[0] = min(
                    column_graph_limits[0],
                    pp[COLUMN]['total portfolio'][MODEL][columns_portfolios].min().min())
                column_graph_limits[1] = max(
                    column_graph_limits[1],
                    pp[COLUMN]['total portfolio'][MODEL][columns_portfolios].max().max())
                
                # mpg.set_xticklabels(
                #     [i for i in pp[COLUMN][TABLE][MODEL].index[xtic]],rotation=30) # changed this to more accurate months
                mpg.tick_params(axis='both', which='major', labelsize=fontsize)
                mpg.tick_params(axis='both', which='minor', labelsize=8)
                mpg.set_title(title,fontsize=fontsize,loc="center")
                
                mpg.grid()
                # fig.show()
                fig.savefig(PATH_PROCESSED.format(
                    date=datestring,model_name=model_name,VARIANT = VARIANT,name_type=TABLE,fileending="png"),
                            dpi=200,bbox_inches = "tight")
        grouped_fig, grouped_axes = plt.subplots(
            n_rows, n_columns, figsize=(6*n_columns, 6*n_rows))
        grouped_fig.suptitle(model_name, fontsize=fontsize)
        
        # important: always only two model specs at the same time.
        # indexing:  
        index = [0,0]
        
        for MODEL in pp[COLUMN]["total portfolio"].keys():
            title = MODEL#+":"+COLUMN.split("_")[1]
            title = title.replace("_"," ")
            index = tuple(index)
            # indexing:
            # model_specs = MODEL.split("_")
            # index = tuple(specs[i].index(j) for i,j in zip([0,1],model_specs))
            
            # index = list(index); index[0]+=1; index=tuple(index);
            # grouped_axes[index].set_xlabel("Time",fontsize=fontsize)
            grouped_axes[index].set_ylabel("Log return",color=color,fontsize=fontsize)
            grouped_axes[index].plot(pp["market"]["total portfolio"]["market_ev"],
                                     color="black", label="market ev",lw=linewidth+1)
            grouped_axes[index].plot(pp["market"]["total portfolio"]["market_vv"],
                                     color="grey", label="market vv",lw=linewidth+1)
            
            for PORTFOLIO in COLUMNS_PORTFOLIOS_DEFAULT:
                grouped_axes[index].plot(pp[COLUMN]["total portfolio"][MODEL][PORTFOLIO],
                                         label=PORTFOLIO.replace("_"," "),lw=linewidth)
            # xtic=[*range(0,PORTFOLIO_PERIOD,48),PORTFOLIO_PERIOD-PORTFOLIO_PERIOD%12]
            # del xtic[-2]
            
            # grouped_axes[index].set_xticks(xtic)
            grouped_axes[index].set_ylim(*[i*1.1 for i in column_graph_limits])
            # grouped_axes[index].set_xticklabels(
            #     [i for i in pp[COLUMN]["total portfolio"][MODEL].index[xtic]],rotation=30) # changed this to more accurate months
            grouped_axes[index].tick_params(axis='both', which='major', labelsize=fontsize)
            grouped_axes[index].tick_params(axis='both', which='minor', labelsize=8)
            grouped_axes[index].set_title(title,fontsize=fontsize,loc="center")
            
            # if index == (0,0):
            #     grouped_axes[index].legend(
            #         prop={'size': fontsize-0.5},
            #         # fancybox=True, shadow=True, 
            #         # bbox_to_anchor=(0., 1.02, 1., .5*HEIGHT_LEGEND+.002), loc='lower left',
            #         bbox_to_anchor=(0, 0.05, 0.25, 1), loc='upper left',
            #         borderaxespad=0.,ncol = 5)
            grouped_axes[index].grid()
            index = list(index)
            index[0]+=1 
            if index[0] == n_rows:
                index = [0,index[1]+1]
        print("Inserting legend...")
        handles, labels = grouped_axes[(1,0)].get_legend_handles_labels()
        
        grouped_fig.legend(
            handles, labels,
            prop={'size': fontsize-0.5},
            # fancybox=True, shadow=True, 
            # bbox_to_anchor=(0., 1.02, 1., .5*HEIGHT_LEGEND+.002), loc='lower left',
            bbox_to_anchor=(0, -0.075, 0.25, 1), loc='lower left',
            borderaxespad=0.,ncol = 5)
        
        grouped_fig.tight_layout()
        grouped_fig.show()
        grouped_fig.savefig(PATH_PROCESSED.format(
            date=datestring,model_name=model_name,VARIANT = VARIANT,name_type=TABLE,fileending="png"),
            dpi=200,bbox_inches = "tight")
            
        # portfolio_performance return
        return pp#, graphs
################################################################################
################################################################################
################################################################################
##########################   Predictive Performance   ##########################
################################################################################
################################################################################
################################################################################

'''
This section is organised in 2 parts:
    
    Volatility Prediction:
        compares all volatility models with each other in terms of predictive performance
        R2, MSE, QLIKE
        
        Over time
        
        Under specific conditions: high volatility, low volatility, maybe semivariance
        
    Return Prediction, SDF
        R2, (QAA)
'''
if False:
    ## predictive performance
    
    pred_p_RP, pred_p_time_RP = prediction_performance(models[:3],"RP", beta_option = False)
    ## RP pred_perf
    rp_R2_stats             = pd.DataFrame(pred_p_RP["res_naive"]["R2_modified"])
    rp_R2_stats["stat"]     = "R2"
    rp_R2_modified          = pd.DataFrame(pred_p_RP["res_naive"]["R2"])
    rp_R2_modified["stat"]  = "R2_modified"
    r2_stats = pd.concat([rp_R2_stats,rp_R2_modified]).reset_index().\
        rename(columns = {"index":"subsample"}).set_index(["stat","subsample"])
    r2_stats*=100
    r2_stats = r2_stats.round(4)
    r2_stats.to_csv(PROCESSED_PATH+"RP_pred_table.csv")
    
    
    pred_p_SDF, pred_p_time_SDF = prediction_performance(models,"SDF", beta_option = False)
    pred_p_VP, pred_p_time_VP = prediction_performance(models[:3],"VP", beta_option = False)

    '''
    Analysis of just 2008 and 2009
    '''
    submodel = "FFNN_wmq_joined"
    
    

    ## portfolio performance


    portfolio_performance_interest_columns = [
        'Mean', 'Std', 'SR annualized', 'adj turnover',
        'Pos ratio', 'Drawdown max', 'beta', 'alpha']
    portfolio_performance_interest_rows = ["HML_ev", "HML_vv","managed"]
    
    portfolios_RP = portfolio_performance(
            models_ana      = models[:3],
            model_name      = "RP",
            years           = YEARS,
            datestring      = "1024", 
            quantiles       = 10,
            )
    RP_perf = {key: table.loc[portfolio_performance_interest_rows,portfolio_performance_interest_columns] 
               for key, table in portfolios_RP["res_naive"]["monthly performance"].items()}
    RP_perf = pd.concat(RP_perf).copy()
    RP_perf[['Mean', 'Std','adj turnover','Pos ratio', 'Drawdown max', 'alpha']]*=100
    RP_perf[['Mean', 'beta', 'alpha']] = \
        RP_perf[['Mean', 'beta', 'alpha']].round(4)
    RP_perf[['Std', 'SR annualized', 'adj turnover', 'Pos ratio', 'Drawdown max']] = \
        RP_perf[['Std', 'SR annualized', 'adj turnover', 'Pos ratio', 'Drawdown max']].round(2)
    RP_perf.to_csv(PROCESSED_PATH+"RP_perf_pred_table.csv")


    portfolios_SDF = portfolio_performance(
            models_ana      = models[:3],
            model_name      = "SDF",
            years           = YEARS,
            datestring      = "1024", 
            quantiles       = 10,
            )
    SDF_perf = {key: table.loc[portfolio_performance_interest_rows,portfolio_performance_interest_columns] 
               for key, table in portfolios_SDF["res_naive"]["monthly performance"].items()}
    SDF_perf = pd.concat(SDF_perf).copy()
    SDF_perf[['Mean', 'Std','adj turnover','Pos ratio', 'Drawdown max', 'alpha']]*=100
    SDF_perf[['Mean', 'beta', 'alpha']] = \
        SDF_perf[['Mean', 'beta', 'alpha']].round(4)
    SDF_perf[['Std', 'SR annualized', 'adj turnover', 'Pos ratio', 'Drawdown max']] = \
        SDF_perf[['Std', 'SR annualized', 'adj turnover', 'Pos ratio', 'Drawdown max']].round(2)
    SDF_perf.to_csv(PROCESSED_PATH+"SDF_perf_pred_table.csv")

    market_perf = portfolios_RP["market"]["monthly performance"]
    market_perf = market_perf[[
        column for column in portfolio_performance_interest_columns if column \
            in market_perf.columns]].copy()
    market_perf[['Mean', 'Std','adj turnover','Pos ratio', 'Drawdown max']]*=100
    market_perf[['Mean']] = \
        market_perf[['Mean']].round(4)
    market_perf[['Std', 'SR annualized', 'adj turnover', 'Pos ratio', 'Drawdown max']] = \
        market_perf[['Std', 'SR annualized', 'adj turnover', 'Pos ratio', 'Drawdown max']].round(2)
    market_perf.to_csv(PROCESSED_PATH+"market_perf_pred_table.csv")

    cosine_similarity = pd.DataFrame({
        "SDF":{model:portfolios_SDF["res_naive"]["monthly performance"][model]["cosine similarity"].values[0]
        for model in ["FFNN_w","FFNN_wmq_joined","FFNN_wmq_mixed"]},
        "RP":{model:portfolios_RP["res_naive"]["monthly performance"][model]["cosine similarity"].values[0]
        for model in ["FFNN_w","FFNN_wmq_joined","FFNN_wmq_mixed"]}}).round(4)

    '''
    Analysis of just 2008 and 2009 for SDF
    '''
    submodel = "FFNN_wmq_joined" # "FFNN_wmq_mixed" # "FFNN_w" # 
    
    data_0809_SDF = portfolios_SDF["res_naive"]["monthly portfolio"][submodel]
    data_0809_SDF = data_0809_SDF.loc[data_0809_SDF["year"].isin([2008,2009]),:]

    performance_0809_SDF = {}
    
    for portfolio_type in ["managed","HML_vv","HML_ev"]:
        performance_0809_SDF[portfolio_type] = {
            "SR":data_0809_SDF[portfolio_type].mean()/data_0809_SDF[portfolio_type].std(),
            "mean":data_0809_SDF[portfolio_type].mean(),
            "std":data_0809_SDF[portfolio_type].std(),
            "SR_annualised":data_0809_SDF[portfolio_type].mean()/data_0809_SDF[portfolio_type].std()*52**.5,
            "TRI":(1+data_0809_SDF[portfolio_type]).prod()-1}


    '''
    Analysis of just 2008 and 2009 for RP
    '''
    data_0809_RP = portfolios_RP["res_naive"]["monthly portfolio"][submodel]
    data_0809_RP = data_0809_RP.loc[data_0809_RP["year"].isin([2008,2009]),:]

    performance_0809_RP = {}
    
    for portfolio_type in ["managed","HML_vv","HML_ev"]:
        performance_0809_RP[portfolio_type] = {
            "SR":data_0809_RP[portfolio_type].mean()/data_0809_RP[portfolio_type].std(),
            "mean":data_0809_RP[portfolio_type].mean(),
            "std":data_0809_RP[portfolio_type].std(),
            "SR_annualised":data_0809_RP[portfolio_type].mean()/data_0809_RP[portfolio_type].std()*52**.5,
            "TRI":(1+data_0809_RP[portfolio_type]).prod()-1}


## prediction performance aggregate


if False:
    models_ana      = models[:3].copy()
    models_ana.append({'name': 'FFNN_w', 'color': 'red', 'ls': 'dotted'})
    model_name      = "VP"
    years           = YEARS

    model_columns = [model["name"] for model in models_ana]

    qlike = Loss_QLikelihood()
    sizes = get_series("mcap")

    monthly_prediction_results = {"R2":{},"MSE":{},"R2_modified":{},"Q-like":{}} # {} # 
    annual_prediction_results = {"R2":{},"MSE":{},"R2_modified":{},"Q-like":{}} # {} # 

    monthly_aggregate = pd.DataFrame()
    annual_aggregate = pd.DataFrame()    

    prediction_results_total = {}

    column = "res_naive"
    index = 0

    for model in models_ana:
        # model = models_ana[0]
        
        name = model["name"]
        if index != 0 and name == "FFNN_w":
            model_name = "VP_HARQ"
        index+=1
        return_data = loader_regressions_P3(
                model_name = name,
                years= YEARS,
                objective_str = model_name,
                path = PATH_PREDICTIONS
                )[["test",*COLUMNS_RESULTS]]
        
        if model_name == "VP_HARQ":
            name = "FFNN_w_HARQ"
        
        return_data = pd.merge(return_data,sizes,left_index=True,right_index=True)
        
        return_data["month"] = return_data.index.get_level_values("date")
        return_data["month"] = return_data["month"].apply(lambda x: x.strftime("%Y-%m"))
        return_data["year"] = return_data["month"].str[:4]
        
        deciles = return_data["mcap"].groupby("date").quantile(.9).to_frame(name="dec10")
        deciles["dec1"] = return_data["mcap"].groupby("date").quantile(.1)
        return_data = pd.merge(return_data,deciles,left_index=True, right_index=True)
        return_data_dec1 = return_data[return_data["mcap"]<return_data["dec1"]]
        return_data_dec10 = return_data[return_data["mcap"]>return_data["dec10"]]
        
        prediction_results_total[name] = {}
        
        ## MSE
        prediction_results_total[name]["MSE"] = {}
        prediction_results_total[name]["MSE"]["Sample"] = ((
           return_data["test"]-return_data["res_naive"])**2).mean()
        
        prediction_results_total[name]["MSE"]["Small"]  = ((
            return_data_dec1["test"]-return_data_dec1["res_naive"])**2).mean()
        prediction_results_total[name]["MSE"]["Large"]  = ((
            return_data_dec10["test"]-return_data_dec10["res_naive"])**2).mean()
        
        ## QLikelihood
        
        prediction_results_total[name]["Q-like"] = {}
        prediction_results_total[name]["Q-like"]["Sample"] = qlike(
           return_data["res_naive"],return_data["test"])
        prediction_results_total[name]["Q-like"]["Small"] = qlike(
            return_data_dec1["res_naive"],return_data_dec1["test"])
        prediction_results_total[name]["Q-like"]["Large"] = qlike(
            return_data_dec10["res_naive"],return_data_dec10["test"],)
        
        
        
        ## R2-pred
        prediction_results_total[name]["R2"] = {}
        prediction_results_total[name]["R2"]["Sample"] = 1-((
           return_data["test"]-return_data["res_naive"])**2).sum()/((
               return_data["test"]-return_data["test"].mean())**2).sum()
        
        prediction_results_total[name]["R2"]["Small"]  = 1-((
            return_data_dec1["test"]-return_data_dec1["res_naive"])**2).sum()/((
                return_data_dec1["test"]-return_data_dec1["test"].mean())**2).sum()
        prediction_results_total[name]["R2"]["Large"]  = 1-((
            return_data_dec10["test"]-return_data_dec10["res_naive"])**2).sum()/((
                return_data_dec10["test"]-return_data_dec10["test"].mean())**2).sum()
        
        
        ## R2-pred modified
        prediction_results_total[name]["R2_modified"] = {}
        prediction_results_total[name]["R2_modified"]["Sample"] = 1-((
           return_data["test"]-return_data["res_naive"])**2).sum()/(return_data["test"]**2).sum()
        
        prediction_results_total[name]["R2_modified"]["Small"]  = 1-((
            return_data_dec1["test"]-return_data_dec1["res_naive"])**2).sum()/(return_data_dec1["test"]**2).sum()
        prediction_results_total[name]["R2_modified"]["Large"]  = 1-((
            return_data_dec10["test"]-return_data_dec10["res_naive"])**2).sum()/(return_data_dec10["test"]**2).sum()        
        
        if name == "FFNN_w":
            monthly_aggregate["RV"] = return_data[["test","month"]].groupby("month").mean()
            annual_aggregate["RV"] = return_data[["test","year"]].groupby("year").mean()
        
        monthly_aggregate[name] = return_data[[column,"month"]].groupby("month").mean()
        annual_aggregate[name] = return_data[[column,"year"]].groupby("year").mean()
        
        ## monthly and annual performance measures
        
        monthly_grouped_predicitons = {x:y for x,y in return_data.groupby("month")}
        annual_grouped_predicitons = {x:y for x,y in return_data.groupby("year")}
        
        for measure in ["R2","MSE","Q-like","R2_modified"]:
            monthly_prediction_results[measure][name] = {}
            annual_prediction_results[measure][name] = {}
        
        for month in monthly_grouped_predicitons.keys():
            monthly_prediction_results["R2"][name][month] = 1-((
                monthly_grouped_predicitons[month]["test"]- monthly_grouped_predicitons[month][column])**2).sum()/((
                    monthly_grouped_predicitons[month]["test"]- monthly_grouped_predicitons[month]["test"].mean())**2).sum()
            
            monthly_prediction_results["R2_modified"][name][month] = 1-((
                monthly_grouped_predicitons[month]["test"]- monthly_grouped_predicitons[month][column])**2).sum()/\
                (monthly_grouped_predicitons[month]["test"]**2).sum()
            
            monthly_prediction_results["Q-like"][name][month] =  qlike(
                monthly_grouped_predicitons[month][column],monthly_grouped_predicitons[month]["test"])
            
            monthly_prediction_results["MSE"][name][month] =  ((
                monthly_grouped_predicitons[month]["test"]-monthly_grouped_predicitons[month][column])**2).mean()
            
            
            monthly_prediction_results[name] = {date:{} for date in monthly_grouped_predicitons.keys()}
            annual_prediction_results[name] = {date:{} for date in annual_grouped_predicitons.keys()}
        
        for year in annual_grouped_predicitons.keys():
            annual_prediction_results["R2"][name][year] = 1-((
                annual_grouped_predicitons[year]["test"]- annual_grouped_predicitons[year][column])**2).sum()/((
                    annual_grouped_predicitons[year]["test"]- annual_grouped_predicitons[year]["test"].mean())**2).sum()
            
            annual_prediction_results["R2_modified"][name][year] = 1-((
                annual_grouped_predicitons[year]["test"]- annual_grouped_predicitons[year][column])**2).sum()/\
                (annual_grouped_predicitons[year]["test"]**2).sum()
            
            annual_prediction_results["Q-like"][name][year] =  qlike(
                annual_grouped_predicitons[year][column],annual_grouped_predicitons[year]["test"])
            
            annual_prediction_results["MSE"][name][year] =  ((
                annual_grouped_predicitons[year]["test"]-annual_grouped_predicitons[year][column])**2).mean()
        
        del monthly_grouped_predicitons, annual_grouped_predicitons

    prediction_results_total = pd.DataFrame.from_dict(
        {(i,j): prediction_results_total[i][j] 
                       for i in prediction_results_total.keys() 
                       for j in prediction_results_total[i].keys()},
                   orient='index')

    stacked_df = prediction_results_total.stack()
    stacked_df = stacked_df.reorder_levels([1, 2, 0])
    reshaped_df = stacked_df.unstack(level=2)
    reshaped_df = reshaped_df.sort_index()
    print(reshaped_df)
    reshaped_df.to_csv(PROCESSED_PATH+"VP_models_perf_pred.csv",sep=",")


    monthly_aggregate.to_csv(PROCESSED_PATH+"VP_models_resid_monthly.csv")
    annual_aggregate.to_csv(PROCESSED_PATH+"VP_models_resid_annual.csv")

    for measure in ["R2","MSE","Q-like","R2_modified"]:
        # measure = "R2"
        
        monthly_df = pd.DataFrame(monthly_prediction_results[measure]).to_csv(PROCESSED_PATH+f"VP_models_{measure}_monthly.csv")
        annual_df = pd.DataFrame(annual_prediction_results[measure]).to_csv(PROCESSED_PATH+f"VP_models_{measure}_annual.csv")


    
## graphs and tables:
if False:
    lead_column = "HARQ"
    model_columns = ["FFNN_w","FFNN_wmq_joined","FFNN_wmq_mixed","FFNN_w_HARQ"]

    ## average predictions over time:
    avg_pred_mtl = pd.merge(
        pd.read_csv(PROCESSED_PATH+"VP_benchmark_resid_monthly.csv", index_col = 0),
            pd.read_csv(PROCESSED_PATH+"VP_models_resid_monthly.csv", index_col = 0)[model_columns],
            left_index = True, right_index = True, how ="inner")
    del avg_pred_mtl["AR1"]
    avg_pred_mtl.rename(inplace=True, columns={
        "FFNN_wmq_joined":"FFNN_wmq", "FFNN_wmq_mixed":"MiFDeL"})
    avg_pred_mtl = np.log(avg_pred_mtl)

    fig, ax = plt.subplots(figsize=(10, 4))

    for column in avg_pred_mtl.columns:
        if column == 'RV':  # replace with the actual column name
            avg_pred_mtl[column].plot(ax=ax, linewidth=3, label=column)  # Highlighted line
        else:
            avg_pred_mtl[column].plot(ax=ax, linewidth=2, label=column)  # Normal lines
    
    # ax.legend()
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=16)
    # ax = avg_pred_mtl.plot(grid=True,legend = False, 
    #                         figsize=(10, 4), fontsize=16)
    # ax.set_title(f"Realised volatility: monthly average prediction", fontsize=16)
    ax.set_ylabel("Predicted Realised Volatility", fontsize=16); ax.set_xlabel("Time", fontsize=16);

    # plt.gca().set_ylim(*value_ranges[metric]) 
    plt.tight_layout()
    # plt.axhline(y=-5, color='red', linestyle='--', label="Cap at -5")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=4, fontsize=16)
    plt.savefig(PROCESSED_PATH+"VP_prediction_monthly.pdf", bbox_inches='tight', dpi=300) 
    plt.show()
    del avg_pred_mtl

    #np.log(avg_pred_mtl).plot(grid=True, legend=True)
    ## legend for this graph needs to be fixed (under graph)   !!!

    avg_pred_ann = pd.merge(
        pd.read_csv(PROCESSED_PATH+"VP_benchmark_resid_annual.csv", index_col = 0),
            pd.read_csv(PROCESSED_PATH+"VP_models_resid_annual.csv", index_col = 0)[model_columns],
            left_index = True, right_index = True, how ="inner")
    del avg_pred_ann["AR1"]
    avg_pred_ann.rename(inplace=True, columns={
        "FFNN_wmq_joined":"FFNN_wmq", "FFNN_wmq_mixed":"MiFDeL"})
    avg_pred_ann = np.log(avg_pred_ann)
   
    fig, ax = plt.subplots(figsize=(10, 4))
   
    for column in avg_pred_ann.columns:
        if column == 'RV':  # replace with the actual column name
            avg_pred_ann[column].plot(ax=ax, linewidth=3, label=column)  # Highlighted line
        else:
            avg_pred_ann[column].plot(ax=ax, linewidth=2, label=column)  # Normal lines
    
    # ax.legend()
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=16)
    # ax = avg_pred_mtl.plot(grid=True,legend = False, 
    #                         figsize=(10, 4), fontsize=16)
    # ax.set_title(f"Realised volatility: monthly average prediction", fontsize=16)
    ax.set_ylabel("Predicted Realised Volatility (annually)", fontsize=16); ax.set_xlabel("Time", fontsize=16);

    # plt.gca().set_ylim(*value_ranges[metric]) 
    plt.tight_layout()
    # plt.axhline(y=-5, color='red', linestyle='--', label="Cap at -5")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=4, fontsize=16)
    plt.savefig(PROCESSED_PATH+"VP_prediction_annual.pdf", bbox_inches='tight', dpi=300) 
    plt.show()
    ## legend for this graph needs to be fixed (under graph)   !!!


    ## table of performance metrics

    total_performance = pd.merge(
        pd.read_csv(PROCESSED_PATH+"VP_benchmark_perf_pred.csv", index_col = [0,1]),
        pd.read_csv(PROCESSED_PATH+"VP_models_perf_pred.csv", index_col = [0,1])[model_columns],
        left_index = True, right_index = True, how ="inner")

    del total_performance["AR1"]

    total_performance.round(3).to_csv(PROCESSED_PATH+f"VP_total_perf_pred.csv",sep=",")

    ## R2 formula: 1-(benchmark-column)/benchmark
    ## MSE and Qlike: 1-(column-benchmark)/bencmark

    total_performance_rel = total_performance.copy()
    total_performance_rel["lead"] = total_performance_rel[lead_column]
    # total_performance_rel.loc[:,"lead"] = total_performance_rel.loc[:,lead_column]

    total_performance_rel1 = total_performance_rel.loc[["R2","R2_modified"]].copy()
    total_performance_rel2 = total_performance_rel.loc[["Q-like","MSE"]].copy()

    for column in total_performance_rel.columns[:-1]:
        # column = "AR1"
        total_performance_rel2.loc[:,column] = 1-(total_performance_rel2["lead"]-total_performance_rel2[column])/total_performance_rel2["lead"]
        total_performance_rel1.loc[:,column] = 1-(total_performance_rel1[column]-total_performance_rel1["lead"])/total_performance_rel1["lead"]

    total_performance_rel = pd.concat([total_performance_rel1,total_performance_rel2])
    del total_performance_rel["lead"]
    total_performance_rel.round(3).to_csv(PROCESSED_PATH+f"VP_total_rel_perf_pred.csv",sep=",")
    
    value_ranges = {
        "R2":(-.5,.5),
        "MSE":(),
        "Q-like":(0,2),
        "R2_modified":(-.5,.5)}
    
    title = {
        "R2":r"$R^2$",
        "MSE":"MSE",
        "Q-like":"Q-likelihood",
        "R2_modified":r"$R^2_*$"}
    
    ## grpah for performance metrics over time
    for metric in ["R2","R2_modified","Q-like","MSE"]:
        # metric = "R2"
        total_performance = pd.merge(
            pd.read_csv(PROCESSED_PATH+f"VP_benchmark_{metric}_monthly.csv", index_col = 0),
            pd.read_csv(PROCESSED_PATH+f"VP_models_{metric}_monthly.csv", index_col = 0)[model_columns],
            left_index = True, right_index = True, how ="inner")
        
        del total_performance["AR1"]
        total_performance.rename(inplace=True, columns={
            "FFNN_wmq_joined":"FFNN_wmq", "FFNN_wmq_mixed":"MiFDeL"})
        # if metric == "Q-like":
        #     total_performance = np.minimum(total_performance, 5)
        
        ax = total_performance.plot(grid=True,legend = False, 
                               figsize=(10, 4), fontsize=16, lw=2)
        ax.set_title(f"{title[metric]} monthly", fontsize=16)
        ax.set_ylabel(f"{title[metric]}", fontsize=16); ax.set_xlabel("time", fontsize=16);
        
        
        if value_ranges[metric] != ():
            plt.gca().set_ylim(*value_ranges[metric]) 
            
        if metric in ["R2", "R2_modified"]:
            ticks = np.linspace(*value_ranges[metric],6)
            custom_labels = [f'<{value_ranges[metric][0]}' if tick == value_ranges[metric][0] \
                             else f'>{value_ranges[metric][1]}' if tick == value_ranges[metric][1] \
                                 else f'{tick:.1f}' for tick in ticks]
            ax.set_yticks(ticks)
            ax.set_yticklabels(custom_labels)
        
        plt.tight_layout()
        # plt.axhline(y=-5, color='red', linestyle='--', label="Cap at -5")
        if metric == "MSE":
            ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=4, fontsize=16)
        plt.savefig(PROCESSED_PATH+f"VP_models_{metric}_monthly.pdf", bbox_inches='tight', dpi=300) 
        plt.show()

        total_performance = pd.merge(
            pd.read_csv(PROCESSED_PATH+f"VP_benchmark_{metric}_annual.csv", index_col = 0),
            pd.read_csv(PROCESSED_PATH+f"VP_models_{metric}_annual.csv", index_col = 0)[model_columns],
            left_index = True, right_index = True, how ="inner")
        
        del total_performance["AR1"]
        total_performance.rename(inplace=True, columns={
            "FFNN_wmq_joined":"FFNN_wmq", "FFNN_wmq_mixed":"MiFDeL"})
        total_performance.reset_index(inplace=True)
        total_performance["index"] = total_performance["index"].astype(str)
        total_performance.set_index("index",inplace=True,drop=True)
        
        ax = total_performance.plot(grid=True,legend = False, 
                               figsize=(10, 4), fontsize=16, lw=2)
        ax.set_title(f"{title[metric]} annually", fontsize=16)
        ax.set_ylabel(f"{title[metric]}", fontsize=16); ax.set_xlabel("time", fontsize=16);
        
        if value_ranges[metric] != ():
            plt.gca().set_ylim(*value_ranges[metric]) 
        if metric in ["R2", "R2_modified"]:
            ticks = np.linspace(*value_ranges[metric],6)
            custom_labels = [f'<{value_ranges[metric][0]}' if tick == value_ranges[metric][0] \
                             else f'>{value_ranges[metric][1]}' if tick == value_ranges[metric][1] \
                                 else f'{tick:.1f}' for tick in ticks]
            ax.set_yticks(ticks)
            ax.set_yticklabels(custom_labels)
        plt.tight_layout()
        # plt.axhline(y=-5, color='red', linestyle='--', label="Cap at -5")
        if metric == "MSE":
            ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=4, fontsize=16)
        plt.savefig(PROCESSED_PATH+f"VP_models_{metric}_annually.pdf", bbox_inches='tight', dpi=300) 
        plt.show()

## predictive performance return prediction
if False:
    ## extract R^_* from prediciton data for small and large stocks as well.
    pass
    
        
##################################################
##################################################
##############  Variable Importance  #############
##################################################
##################################################

if False:
    #########################
    # Preparing the dataset #
    #########################
    
    ## from ELK_NETWORKS.ELK_dataset use the following constants:
    label_file = "labels{frequency:s}.csv"
    data_identification_file = "ident{frequency:s}.csv"
    x_name = "data_b{batch:d}x{frequency:s}.h5"
    y_name = "data_b{batch:d}y1.h5"
    
    datasets_f = {}
    frequencies = ["weekly","monthly","quarterly"]
    
    ## import identification gile
    data_identification = \
        {frequency_f: pd.read_csv(PATH_DATA+data_identification_file.format(frequency = frequency_f[0]),
                                  sep=";",index_col=0) for frequency_f in frequencies}
    data_identification["weekly"] = data_identification["weekly"][
        data_identification["weekly"]["target_date"]>="2003-01-01"]
    data_identification["monthly"] = data_identification["monthly"][
        data_identification["monthly"]["target_date"]>="2002-11-01"]
    data_identification["quarterly"] = data_identification["quarterly"][
        data_identification["quarterly"]["target_date"]>="2002-08-01"]
    
    ## import label files
    labels = {frequency_f: pd.read_csv(PATH_DATA+label_file.format(frequency = frequency_f[0]),
                                       sep=";",index_col=0)["name"].tolist() 
              for frequency_f in frequencies}
    
    
    for frequency_f in frequencies:
        frequency = frequency_f[0]
        
        if frequency_f == "weekly":
            dataset_y = pd.concat([
                pd.read_hdf(PATH_DATA+y_name.format(batch=batch),key="data") 
                for batch in data_identification[frequency_f]["batch"]])
        
        datasets_f[frequency_f] = pd.concat([
            pd.read_hdf(PATH_DATA+x_name.format(batch=batch, frequency = frequency),key="data") 
            for batch in data_identification[frequency_f]["batch"]])
        
    ## merging
    ## first, merge weekly data with linking table
    dataset = pd.merge(datasets_f["weekly"],dataset_y[["w_date","m_date","q_date"]],
                       left_index=True,right_index=True,how="inner")
    
    ## then merge it with monthly data
    dataset = pd.merge(dataset,datasets_f["monthly"],
                       left_on=["gvkey","m_date"],right_index=True,how="inner")
    
    ## then merge it with quarterly data
    dataset = pd.merge(dataset,datasets_f["quarterly"],
                       left_on=["gvkey","q_date"],right_index=True,how="inner")
    del datasets_f, dataset["w_date"], dataset["m_date"], dataset["q_date"]

    ## na handling:
    sample_avg = dataset.groupby("date").mean() 
    columns = dataset.columns.copy()
    sample_avg.rename(columns = {column:column+"_avg" for column in columns},inplace=True)
    dataset.reset_index(inplace=True)
    dataset = pd.merge(dataset,sample_avg, left_on="date", right_on ="date",how="inner")
    dataset.set_index(["date","gvkey"],inplace=True)
    
    for column in columns:
        dataset.loc[dataset[column].isna(),column]=dataset.loc[dataset[column].isna(),column+"_avg"]
        del dataset[column+"_avg"]
        
    dataset.replace(np.nan,0,inplace=True)
    dataset.to_hdf(PATH_DATA_RAW+f"dataset_VI_{variant}.h5",key="data")
    dataset_y[["mcap","FSS2","q_dist","m_dist","ret1w","rvarf1w"]].to_hdf(PATH_DATA_RAW+f"dataset_VI_{variant}_y.h5",key="data")
    
  
class numpy_df_index_handler():
    def __init__(self,df_columns):
        self.df_columns = np.array(df_columns)
        
    def __getitem__(self,columns):
        
        if isinstance(columns, list):
            return [np.where(self.df_columns == column)[0][0] for column in columns]
        return np.where(self.df_columns == columns)[0][0]
        
columns_prediction = {
    "HAR":['rvar1w', 'rvar4w', 'rvar13w'],
    "HARQ":['rvar1w', 'rvar4w', 'rvar13w', 'rq1w'],
    "HARQ-F":['rvar1w', 'rvar4w', 'rvar13w', 'rq1w', 'rq4w', 'rq13w'],
    "w":[
        'b', 'b2', 'bas', 'dso4w', 'ill', 'im1w', 'im26w', 'imcap', 'irv26w', 
        'irv156w', 'mcap', 'mdr1w', 'mdr4w', 'sto1w', 'tv1w', 'vst1w', 'vtv1w',
        'sto4w', 'tv4w', 'vst4w', 'vtv4w', 'ztd1w', 'm1w', 'm4w', 'm13w',
        'm26w', 'rvar1w', 'rvar4w', 'rvar13w', 'rvar26w', 'EAR', 'EAV', 
        'irvar13w', 'itv1w', 'itv4w', 'imdr4w', 'ib', 'iirv26w', 'rq1w', 'rq4w',
        'rq13w'],
    "wmq":[
        'b', 'b2', 'bas', 'dso4w', 'ill', 'im1w', 'im26w', 'imcap', 'irv26w', 
        'irv156w', 'mcap', 'mdr1w', 'mdr4w', 'sto1w', 'tv1w', 'vst1w', 'vtv1w', 
        'sto4w', 'tv4w', 'vst4w', 'vtv4w', 'ztd1w', 'm1w', 'm4w', 'm13w', 
        'm26w', 'rvar1w', 'rvar4w', 'rvar13w', 'rvar26w', 'EAR', 'EAV', 
        'irvar13w', 'itv1w', 'itv4w', 'imdr4w', 'ib', 'iirv26w', 'rq1w', 'rq4w', 
        'rq13w', 'BM', 'CFP', 'DIVP', 'DM6M', 'EP', 'IBM', 'ICFP', 'IM6M',
        'IM12M', 'IMCAP', 'M1M', 'M6M', 'M12M', 'M36M', 'MCAP', 'SP', 'RVAR1M', 
        'RVAR6M', 'DEP1M', 'DEP12M', 'IEP', 'DCFP1M', 'DCFP12M', 'DSP1M', 
        'DSP12M', 'ISP', 'ACA', 'ACV', 'ACW', 'AG1Y', 'C', 'CEI', 'CFD', 'CFV', 
        'CIN', 'CP', 'CR', 'DCE', 'DCR', 'DCSE', 'DD', 'DE', 'DGMDS', 'DI', 
        'DLTD', 'DNOA', 'DPPE', 'DQR', 'DRD', 'DS', 'DSDAR', 'DSDI', 'DSDSGA', 
        'DSI', 'DTAX', 'EIS', 'EV', 'FSS', 'FSS2', 'GP', 'IDAT', 'IDCE', 'IDE', 
        'IDPM', 'ISCH', 'L', 'OC', 'OP', 'QR', 'RDS', 'RE', 'ROA', 'ROE', 'ROI', 
        'RS', 'SC', 'SD', 'SI', 'SR', 'TANG', 'TIBI', 'DEAR', 'RDE', 'RDC', 'SGAE', 
        'ARI', 'DSGAE', 'CFRD', 'DACA', 'DROI', 'DROE', 'DRDS', 'DFSS', 'IACA', 
        'IROI', 'IROE', 'IOP', 'IRDS', 'IDS', 'IFSS', 'IDEAR', 'ISS']}
    
new_variables = {
    "w":['sto1w', 'tv1w', 'vst1w', 'vtv1w','irvar13w', 'itv1w', 'itv4w', 'imdr4w', 
         'ib', 'iirv26w',"ill","im1w","irv156w","mdr1w", ],
    "wmq":['sto1w', 'tv1w', 'vst1w', 'vtv1w','irvar13w', 'itv1w', 'itv4w', 'imdr4w', 
           'ib', 'iirv26w',"ill","im1w","irv156w","mdr1w", "RVAR6M", "DEP1M", 
           "DEP12M", "IEP", "DCFP1M", "DCFP12M", "DSP1M", "DSP12M", "ISP",
           'DEAR', 'RDE', 'RDC', 'SGAE', 'ARI', 'DSGAE', 'CFRD', 'DACA', 'DROI',
           'DROE', 'DRDS', 'DFSS', 'IACA', 'IROI', 'IROE', 'IOP', 'IRDS', 'IDS',
           'IFSS', 'IDEAR', 'ISS']}

def vi_measure(
        dataset, ## dataset and dataset_y need to be precleaned.
        dataset_y,
        column_set = "HAR", 
        absolute=False,
        relative = False):
    # import pdb; pdb.set_trace();
    common_index = dataset.index.intersection(dataset_y.index)
    dataset_l = dataset.reindex(common_index).copy() 
    dataset_y_l = dataset_y.reindex(common_index).copy()
    
    columns = columns_prediction[column_set]
    
    scaler = StandardScaler()
    analysis_dataset = scaler.fit_transform(dataset_l[columns])

    pca_matrix = PCA(n_components=len(columns))
    
    principal_components = pca_matrix.fit_transform(analysis_dataset) 
    
    df_pca = pd.DataFrame(data=principal_components, columns=[str(i) for i in range(1,len(columns)+1)])
    
    df_pca["target"] = dataset_y_l.values
    
    correlation = df_pca.corr()["target"].values[:-1]
    del df_pca["target"]
    
    eigenvalues = pca_matrix.explained_variance_ratio_
    
    impact = np.matmul(pca_matrix.components_,correlation)
    
    impact_direction = (impact>0)
    
    if absolute:
        impact = abs(impact)
    if relative:
        impact/=sum(abs(impact))
        
    return impact, impact_direction, eigenvalues, correlation/abs(correlation).sum(), pca_matrix.components_
    
def vector_similarity(a,b):
    a_b = sum(a*b)
    a_len = sum(a*a)**.5
    b_len = sum(b*b)**.5
    cos_sim = a_b/(a_len*b_len)
    length_ratio = a_len/b_len
    dist = sum((a-b)**2)**.5
    angle = np.arccos(cos_sim)/(math.pi/2)*90
    length_corrected_dist = sum((a-b*length_ratio)**2)**.5
    return {
        "cosine similarity":cos_sim, "length ratio":length_ratio,
        "angle":angle,"euclidean distance":dist, 
        "euclidean distance (length corrected)":length_corrected_dist}
    


from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
def add_colorbar_box(cbar,fig, n_boxes=1):
    pos = cbar.ax.get_position()  # Get position of the colorbar
    box = FancyBboxPatch((pos.x0 +.01-(n_boxes-1)*.01, pos.y0 - 0.015+(n_boxes-1)*.01),  # x, y position #
                         pos.width + 0.13-(n_boxes-1)*.01,  # width
                         pos.height + 0.03 - (n_boxes-1)*.03,  # height
                         boxstyle="round,pad=0.02",  # Rounded box with slight padding
                         edgecolor="black",  # Border color
                         linewidth=1.5,  # Border thickness
                         facecolor="none",  # Transparent fill
                         zorder=10)  # Ensures box appears above other elements
    fig.add_artist(box)  # Add the box to the figure


def plot_VI(VI, mode = "single",
            fontsize = 30,
            display = 10,
            datestring = "1024",
            objective = "RP",
            name = "Top 10 RP wmq"):
    """
    plots variable importance given a dataframe containing the actual importance and 
        models prediction importance.

    Parameters
    ----------
    VI : TYPE
        DESCRIPTION.
    mode : TYPE, optional
        DESCRIPTION. The default is "single".
    fontsize : TYPE, optional
        DESCRIPTION. The default is 18.
    display : TYPE, optional
        DESCRIPTION. The default is 10.
    datestring : TYPE, optional
        DESCRIPTION. The default is "1024".
    objective : TYPE, optional
        DESCRIPTION. The default is "RP".

    Returns
    -------
    None.

    Testing Parameters
    ------------------
        VI          = FFNN_wmq_VP_VI.copy()
        fontsize    = 30
        display     = 10
        datestring  = "1024"
        objective   = "RP"
        name        = ["Top 10 model RP wmq","Top 10 actual RP wmq", "Highest diff RP wmq"]
        mode        = "multi"
        
        name        = ["HARQ-F VP"]
        VI          = HARQF_VI.copy()
        mode        = "single"
    """
    if type(name) == str:
        name = [name]
    # VI          = HARQF_VI.copy()
    # VI          = FFNN_wmq_VP_VI.copy()
    # VI          = HAR_VI.copy()
    # VI          = FFNN_wmq_SDF_VI
    # VI          = FFNN_w_SDF_VI
    VI = VI.copy()
    
    present_columns = [column for column in VI.columns if column not in [
        "category","abs","diff"]]
    
    v_max = max(VI[present_columns].abs().max())
    
    if "category" in VI.columns:
        VI['Frequency'] = VI['category'].map({'w': 2, 'm': 3, 'q': 4})
        present_columns.append("Frequency")
        
    ## color map heatmap
    cmap1 = plt.get_cmap("bwr").copy() # "coolwarm", 'seismic'
    cmap1.set_over('none')
    
    
    fig, ax = plt.subplots(figsize=(10,6)) # 10*(2+(5/4)*len(present_columns))/7, 6*(3+display)/(13)))
    sns.heatmap(VI[present_columns].head(10),  ax = ax,
                # annot=True, fmt=".2f", annot_kws={"size": 8, "color": "black"},
                linewidths=.5, vmin=-v_max, vmax = v_max, cmap=cmap1, cbar = True)
    plt.tight_layout()

    cbar1 = ax.collections[0].colorbar  
    cbar1.set_label('Exposure Total %', fontsize=fontsize-5)#, labelpad=10)
    cbar1.ax.tick_params(labelsize=fontsize-5)#, rotation = 30)
    cbar1.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
    if len(present_columns) == 3:
        adjustment_index = .09 
    elif len(present_columns) == 2:
        adjustment_index = .09
    elif len(present_columns) == 1:
        adjustment_index = .05
    else:
        adjustment_index = .05
    cbar1.ax.set_position([-0.3-adjustment_index, -.025, 0.05, 1])  # Adjust position [left, bottom, width, height]
    
    ## if categories are fed, this is what happens:
    if "category" in VI.columns:
        category_colors = ['#009E73', '#E69F00', '#CC79A7']
        cmap2 = mcolors.ListedColormap(category_colors)
        cmap2.set_under("none")
        sns.heatmap(VI[present_columns].head(10),  ax = ax, annot=False,
                    linewidths=.5, vmin=2, vmax=4, cmap=cmap2, cbar=True)
    
        cbar2 = ax.collections[1].colorbar  
    
        cbar1.ax.set_position([-0.3-adjustment_index, 0.475, 0.05, 0.5])  # Adjust position [left, bottom, width, height]
        cbar2.ax.set_position([-0.3-adjustment_index, -0.075, 0.05, 0.5])  # Adjust position [left, bottom, width, height]
        
        cbar2.set_ticks([2.33, 3, 3.66])  # Adjusted tick positions for categories
        cbar2.set_ticklabels(['w', 'm', 'q'],fontsize = fontsize-5) ## set category labels
        cbar2.set_label("Frequency", fontsize=fontsize-5) # set label of colorbar 2
        add_colorbar_box(cbar2, fig, 2)
        add_colorbar_box(cbar1, fig, 2)
    else:
        add_colorbar_box(cbar1,fig)
    # Set title and adjust layout
    xlabel = "Series"; ylabel = "Variables"
    ax.set_xlabel(xlabel,fontsize=fontsize); ax.set_ylabel(ylabel,fontsize=fontsize);
    ax.tick_params(axis='both', which='major', labelsize=fontsize-5, rotation = 30)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize-5, rotation = 30)
    ax.set_title(f"{name[0]}", fontsize=fontsize)
    plt.show()
    fig.get_figure().savefig(PROCESSED_PATH+f"VI_{objective}_{name[0]}.pdf",
                             dpi=200,bbox_inches = "tight")
    
    if mode != "single":
        # print more graphs here but without legend.
        fig, ax = plt.subplots(figsize=(6,6)) # 6*(len(present_columns)/4), 6*(3+display)/(13)))
        sns.heatmap(VI.sort_values("actual",key=abs,ascending=False)[present_columns].head(10),  ax = ax,
                    linewidths=.5, vmin=-v_max, vmax = v_max, cmap=cmap1, cbar = False)
        plt.tight_layout()

        ## if categories are fed, this is what happens:
        if "category" in VI.columns:
            category_colors = ['#009E73', '#E69F00', '#CC79A7']
            cmap2 = mcolors.ListedColormap(category_colors)
            cmap2.set_under("none")
            sns.heatmap(VI.sort_values("actual",key=abs,ascending=False)[present_columns].head(10),  ax = ax, 
                        linewidths=.5, vmin=2, vmax=4, cmap=cmap2, cbar=False)
        
        # Set title and adjust layout
        xlabel = "Series"; ylabel = "Variables"
        ax.set_xlabel(xlabel,fontsize=fontsize); ax.set_ylabel(ylabel,fontsize=fontsize);
        ax.tick_params(axis='both', which='major', labelsize=fontsize-5, rotation = 30)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-5, rotation = 30)
        ax.set_title(f"{name[1]}", fontsize=fontsize)
        plt.show()
        fig.get_figure().savefig(PROCESSED_PATH+f"VI_{objective}_{name[1]}.pdf",
                                 dpi=200,bbox_inches = "tight")
        
        
        
        fig, ax = plt.subplots(figsize=(6,6)) # 6*(len(present_columns)/4), 6*(3+display)/(13)))
        sns.heatmap(VI.sort_values("diff",key=abs,ascending=False)[present_columns].head(10),  ax = ax,
                    linewidths=.5, vmin=-v_max, vmax = v_max, cmap=cmap1, cbar = False)
        plt.tight_layout()

        ## if categories are fed, this is what happens:
        if "category" in VI.columns:
            category_colors = ['#009E73', '#E69F00', '#CC79A7']
            cmap2 = mcolors.ListedColormap(category_colors)
            cmap2.set_under("none")
            sns.heatmap(VI.sort_values("diff",key=abs,ascending=False)[present_columns].head(10),  ax = ax, 
                        linewidths=.5, vmin=2, vmax=4, cmap=cmap2, cbar=False)
        
        # Set title and adjust layout
        xlabel = "Series"; ylabel = "Variables"
        ax.set_xlabel(xlabel,fontsize=fontsize); ax.set_ylabel(ylabel,fontsize=fontsize);
        ax.tick_params(axis='both', which='major', labelsize=fontsize-5, rotation = 30)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize-5, rotation = 30)
        ax.set_title(f"{name[2]}", fontsize=fontsize)
        plt.show()
        fig.get_figure().savefig(PROCESSED_PATH+f"VI_{objective}_{name[2]}.pdf",
                                 dpi=200,bbox_inches = "tight")

if False:
    #################
    #  Perform PCA  #
    #################
    
    
    ## load data
    dataset = pd.read_hdf(PATH_DATA_RAW+f"dataset_VI_{variant}.h5",key="data")
    dataset = dataset.reset_index().set_index(["gvkey","date"])
    dataset_y = pd.read_hdf(PATH_DATA_RAW+f"dataset_VI_{variant}_y.h5",key="data")
    dataset.sort_values(["gvkey","date"],inplace=True)
    dataset_y.sort_values(["gvkey","date"],inplace=True)
    
    if False:
        factor_decile_portfolios = {}
        dataset["ret1w"] = dataset_y["ret1w"]
        factor_decile_portfolios["market"] = dataset.groupby("date")["ret1w"].mean()
        for column in columns_prediction["wmq"]:
            # column = "m1w"
            quantiles = pd.DataFrame({
                "q10": dataset.groupby("date")[column].quantile(.9),
                "q1" : dataset.groupby("date")[column].quantile(.1)})
            column_portfolio = pd.merge(
                dataset[[column,"ret1w"]].reset_index(), quantiles, 
                left_on = "date", right_on = "date", how="inner")
            factor_decile_portfolios[column+"_up"] = column_portfolio[
                column_portfolio[column]>=column_portfolio["q10"]].\
                groupby("date")["ret1w"].mean()
            factor_decile_portfolios[column+"_down"] = column_portfolio[
                column_portfolio[column]<=column_portfolio["q1"]].\
                groupby("date")["ret1w"].mean()
        del dataset["ret1w"]
        factor_decile_portfolios = pd.DataFrame(factor_decile_portfolios)
        factor_decile_performance = pd.DataFrame({
            "mean_return":factor_decile_portfolios.mean(),
            "std_return":factor_decile_portfolios.std(),
            "total_return":(1+factor_decile_portfolios).prod()})
        factor_decile_performance["SR"] = factor_decile_performance["mean_return"]/\
            factor_decile_performance["std_return"]
        factor_decile_performance.sort_values("SR",inplace=True,ascending=False)
        
    
    impact_VP = {
        column_set:vi_measure(dataset, dataset_y["rvarf1w"], column_set)
        for column_set in columns_prediction.keys()}
    
    impact_RP = {
        column_set:vi_measure(dataset, dataset_y["ret1w"], column_set)
        for column_set in columns_prediction.keys()}
    
    ## models VI:
    
    ## HAR, HARQ, HARQ-F
    predictions_VP_benchmarks = pd.read_hdf(
        BASE_PATH +"Paper3 results/Iterations/Benchmarks/VP_TS_pred.h5",key="data")
    
    impact_VP_models = {
        b_model:vi_measure(dataset, predictions_VP_benchmarks[b_model], b_model)
        for b_model in ["HAR","HARQ","HARQ-F"]}
    
    impact_similarity_VP = {
        b_model: vector_similarity(impact_VP_models[b_model][0],impact_VP[b_model][0])
        for b_model in  ["HAR","HARQ","HARQ-F"]}
    
    ## ML models:
    
        
    ## VP
    names = {"FFNN_w":"FFNN_w","FFNN_wmq_joined":"FFNN_wmq","FFNN_wmq_mixed":"MiFDeL",
             "VP_HARQ":"FFNN_w_HARQ"}
    predictions_VP ={}
    for model_name in ["FFNN_w","FFNN_wmq_joined","FFNN_wmq_mixed","VP_HARQ"]:
        
        predictions_VP[names[model_name]] =  loader_regressions_P3(
                model_name = model_name if model_name != "VP_HARQ" else "FFNN_w",
                years= YEARS,
                objective_str = "VP" if model_name != "VP_HARQ" else "VP_HARQ",
                path = PATH_PREDICTIONS
                )["res_naive"]
    predictions_VP = pd.concat(predictions_VP,axis=1)
    
    impact_VP_models["FFNN_w"] = vi_measure(dataset, predictions_VP["FFNN_w"], "w")
    impact_VP_models["FFNN_wmq"] = vi_measure(dataset, predictions_VP["FFNN_wmq"], "wmq")
    impact_VP_models["MiFDeL"] = vi_measure(dataset, predictions_VP["MiFDeL"], "wmq")
    impact_VP_models["FFNN_w_HARQ"] = vi_measure(dataset, predictions_VP["FFNN_w_HARQ"], "HARQ-F")
    
    impact_similarity_VP["FFNN_w"] =  vector_similarity(impact_VP_models["FFNN_w"][0],impact_VP["w"][0])
    impact_similarity_VP["FFNN_wmq"] = vector_similarity(impact_VP_models["FFNN_wmq"][0],impact_VP["wmq"][0])
    impact_similarity_VP["MiFDeL"] = vector_similarity(impact_VP_models["MiFDeL"][0],impact_VP["wmq"][0])
    impact_similarity_VP["FFNN_w_HARQ"] = vector_similarity(impact_VP_models["FFNN_w_HARQ"][0],impact_VP["HARQ-F"][0])
    
    ## make tables comparing the vectors
    impact_similarity_VP = pd.DataFrame(impact_similarity_VP)
    impact_similarity_VP.round(4).to_csv(PROCESSED_PATH+"VI_VP_sim_tab.csv",sep=";")
    
    ## RP
    predictions_RP ={}
    for model_name in ["FFNN_w","FFNN_wmq_joined","FFNN_wmq_mixed"]:
        
        predictions_RP[names[model_name]] =  loader_regressions_P3(
                model_name = model_name,
                years= YEARS,
                objective_str = "RP",
                path = PATH_PREDICTIONS
                )["res_naive"]
    predictions_RP = pd.concat(predictions_RP,axis=1)
    
    impact_RP_models = {}
    impact_RP_models["FFNN_w"] = vi_measure(dataset, predictions_RP["FFNN_w"], "w")
    impact_RP_models["FFNN_wmq"] = vi_measure(dataset, predictions_RP["FFNN_wmq"], "wmq")
    impact_RP_models["MiFDeL"] = vi_measure(dataset, predictions_RP["MiFDeL"], "wmq")
    
    impact_similarity_RP = {}
    impact_similarity_RP["FFNN_w"] =  vector_similarity(impact_RP_models["FFNN_w"][0],impact_RP["w"][0])
    impact_similarity_RP["FFNN_wmq"] = vector_similarity(impact_RP_models["FFNN_wmq"][0],impact_RP["wmq"][0])
    impact_similarity_RP["MiFDeL"] = vector_similarity(impact_RP_models["MiFDeL"][0],impact_RP["wmq"][0])
    impact_similarity_RP = pd.DataFrame(impact_similarity_RP)
    impact_similarity_RP.round(4).to_csv(PROCESSED_PATH+"VI_RP_sim_tab.csv",sep=";")
    
    ##SDF
    predictions_SDF ={}
    for model_name in ["FFNN_w","FFNN_wmq_joined","FFNN_wmq_mixed"]:
        
        predictions_SDF[names[model_name]] =  loader_regressions_P3(
                model_name = model_name,
                years= YEARS,
                objective_str = "SDF",
                path = PATH_PREDICTIONS
                )["res_naive"]
    predictions_SDF = pd.concat(predictions_SDF,axis=1)
    
    impact_SDF_models = {}
    impact_SDF_models["FFNN_w"] = vi_measure(dataset, predictions_SDF["FFNN_w"], "w")
    impact_SDF_models["FFNN_wmq"] = vi_measure(dataset, predictions_SDF["FFNN_wmq"], "wmq")
    impact_SDF_models["MiFDeL"] = vi_measure(dataset, predictions_SDF["MiFDeL"], "wmq")
    
    
    
    
    ## make heatmap about top most important variables, and highest difference in influence
    
    HAR_VI = pd.DataFrame({
        "actual":impact_VP["HAR"][0],
        "HAR":impact_VP_models["HAR"][0]},
        index = columns_prediction["HAR"])
    HAR_VI/=abs(HAR_VI).sum(axis=0)
    HAR_VI["abs"] = HAR_VI["HAR"].abs()
    HAR_VI.sort_values("abs",ascending=False,inplace=True)
    HAR_VI["diff"] = HAR_VI["actual"] - HAR_VI["HAR"]
    # del HAR_VI["abs"]
    plot_VI(HAR_VI, mode="single",display = 3,objective="VP", name="HAR VP")
    
    HARQ_VI = pd.DataFrame({
        "actual":impact_VP["HARQ"][0],
        "HARQ":impact_VP_models["HARQ"][0]},
        index = columns_prediction["HARQ"])
    HARQ_VI/=abs(HARQ_VI).sum(axis=0)
    HARQ_VI["abs"] = HARQ_VI["HARQ"].abs()
    HARQ_VI.sort_values("abs",ascending=False,inplace=True)
    HARQ_VI["diff"] = HARQ_VI["actual"] - HARQ_VI["HARQ"]
    # del HARQ_VI["abs"]
    plot_VI(HARQ_VI, mode="single",display = 4,objective="VP", name="HARQ VP")
    
    HARQF_VI = pd.DataFrame({
        "actual":impact_VP["HARQ-F"][0],
        "HARQ-F":impact_VP_models["HARQ-F"][0],
        "FFNN_w_HARQ":impact_VP_models["FFNN_w_HARQ"][0],},
        index = columns_prediction["HARQ-F"])
    HARQF_VI/=abs(HARQF_VI).sum(axis=0)
    HARQF_VI["abs"] = (HARQF_VI["HARQ-F"].abs() + \
                       HARQF_VI["FFNN_w_HARQ"].abs())*.5
    HARQF_VI.sort_values("abs",ascending=False,inplace=True)
    HARQF_VI["diff"] = HARQF_VI["actual"] - HARQF_VI["FFNN_w_HARQ"]
    # del HARQF_VI["abs"]
    plot_VI(HARQF_VI, mode="single",display = 4,objective="VP", name="HARQ-F VP")
    
    FFNN_w_VP_VI = pd.DataFrame({
        "actual":impact_VP["w"][0],
        "FFNN_w":impact_VP_models["FFNN_w"][0]},
        index = columns_prediction["w"])
    FFNN_w_VP_VI/=abs(FFNN_w_VP_VI).sum(axis=0)
    FFNN_w_VP_VI["abs"] = FFNN_w_VP_VI["FFNN_w"].abs()
    FFNN_w_VP_VI.sort_values("abs",ascending=False,inplace=True)
    FFNN_w_VP_VI["diff"] = FFNN_w_VP_VI["actual"] - FFNN_w_VP_VI["FFNN_w"]
    # del FFNN_w_VP_VI["abs"]
    plot_VI(FFNN_w_VP_VI, mode="multi",display = 10,objective="VP", 
            name=["Top 10 w model", "Top 10 w actual","Top 10 (actual-model)"])
    
    FFNN_wmq_VP_VI = pd.DataFrame({
        "actual":impact_VP["wmq"][0],
        "FFNN_wmq":impact_VP_models["FFNN_wmq"][0],
        "MiFDeL":impact_VP_models["MiFDeL"][0]},
        index = columns_prediction["wmq"])
    FFNN_wmq_VP_VI/=abs(FFNN_wmq_VP_VI).sum(axis=0)
    FFNN_wmq_VP_VI["abs"] = (FFNN_wmq_VP_VI["FFNN_wmq"].abs() + \
                             FFNN_wmq_VP_VI["MiFDeL"].abs())*.5
    FFNN_wmq_VP_VI["diff"] = FFNN_wmq_VP_VI["actual"] - (
        FFNN_wmq_VP_VI["FFNN_wmq"]+FFNN_wmq_VP_VI["FFNN_wmq"])/2
    
    ## time importance VP
    FFNN_wmq_VP_VI_category = abs(FFNN_wmq_VP_VI).copy()
    FFNN_wmq_VP_VI_category["category"] = [*["w"]*41,*["m"]*26,*["q"]*76]
    FFNN_wmq_VP_VI_category = FFNN_wmq_VP_VI_category.groupby("category").sum()
    FFNN_wmq_VP_VI_category.sort_values("abs",ascending=False,inplace=True)
    del FFNN_wmq_VP_VI_category["abs"]
    
    FFNN_wmq_VP_VI["category"] = [*["w"]*41,*["m"]*26,*["q"]*76]
    FFNN_wmq_VP_VI.sort_values("abs",ascending=False,inplace=True)
    
    plot_VI(FFNN_wmq_VP_VI, mode="multi",display = 10,objective="VP", 
            name=["Top 10 wmq model", "Top 10 wmq actual","Top 10 wmq (actual-model)"])
    
    ## Variable importance RP
    
    FFNN_w_RP_VI = pd.DataFrame({
        "actual":impact_RP["w"][0],
        "FFNN_w":impact_RP_models["FFNN_w"][0]},
        index = columns_prediction["w"])
    FFNN_w_RP_VI/=abs(FFNN_w_RP_VI).sum(axis=0)
    FFNN_w_RP_VI["abs"] = FFNN_w_RP_VI["FFNN_w"].abs()
    FFNN_w_RP_VI.sort_values("abs",ascending=False,inplace=True)
    FFNN_w_RP_VI["diff"] = FFNN_w_RP_VI["actual"] - FFNN_w_RP_VI["FFNN_w"]
    # del FFNN_w_RP_VI["abs"]
    plot_VI(FFNN_w_RP_VI, mode="multi",display = 10,objective="RP", 
            name=["Top 10 w model", "Top 10 w actual","Top 10 (actual-model)"])
    
    FFNN_wmq_RP_VI = pd.DataFrame({
        "actual":impact_RP["wmq"][0],
        "FFNN_wmq":impact_RP_models["FFNN_wmq"][0],
        "MiFDeL":impact_RP_models["MiFDeL"][0]},
        index = columns_prediction["wmq"])
    FFNN_wmq_RP_VI/=abs(FFNN_wmq_RP_VI).sum(axis=0)
    FFNN_wmq_RP_VI["abs"] = (FFNN_wmq_RP_VI["FFNN_wmq"].abs() + \
                             FFNN_wmq_RP_VI["MiFDeL"].abs())*.5
    FFNN_wmq_RP_VI["diff"] = FFNN_wmq_RP_VI["actual"] - (
        FFNN_wmq_RP_VI["FFNN_wmq"]+FFNN_wmq_RP_VI["FFNN_wmq"])/2
    
    ## time importance VP
    FFNN_wmq_RP_VI_category = abs(FFNN_wmq_RP_VI).copy()
    FFNN_wmq_RP_VI_category["category"] = [*["w"]*41,*["m"]*26,*["q"]*76]
    FFNN_wmq_RP_VI_category = FFNN_wmq_RP_VI_category.groupby("category").sum()
    FFNN_wmq_RP_VI_category.sort_values("abs",ascending=False,inplace=True)
    # del FFNN_wmq_RP_VI_category["abs"]
    
    FFNN_wmq_RP_VI["category"] = [*["w"]*41,*["m"]*26,*["q"]*76]
    FFNN_wmq_RP_VI.sort_values("abs",ascending=False,inplace=True)
    plot_VI(FFNN_wmq_RP_VI, mode="multi",display = 10,objective="RP", 
            name=["Top 10 wmq model", "Top 10 wmq actual","Top 10 wmq (actual-model)"])
    # del FFNN_wmq_RP_VI["abs"]
    
    
    
    ## Variable importance SDF
    
    FFNN_w_SDF_VI = pd.DataFrame({
        "FFNN_w":impact_SDF_models["FFNN_w"][0]},
        index = columns_prediction["w"])
    FFNN_w_SDF_VI/=abs(FFNN_w_SDF_VI).sum(axis=0)
    FFNN_w_SDF_VI["abs"] = FFNN_w_SDF_VI["FFNN_w"].abs()
    FFNN_w_SDF_VI.sort_values("abs",ascending=False,inplace=True)
    # del FFNN_w_VP_RP["abs"]
    plot_VI(FFNN_w_SDF_VI, mode="single",display = 10,objective="SDF", 
            name=["Top 10 w SDF model"])
    
    FFNN_wmq_SDF_VI = pd.DataFrame({
        "FFNN_wmq":impact_SDF_models["FFNN_wmq"][0],
        "MiFDeL":impact_SDF_models["MiFDeL"][0]},
        index = columns_prediction["wmq"])
    FFNN_wmq_SDF_VI/=abs(FFNN_wmq_SDF_VI).sum(axis=0)
    FFNN_wmq_SDF_VI["abs"] = (FFNN_wmq_SDF_VI["FFNN_wmq"].abs() + \
                             FFNN_wmq_SDF_VI["MiFDeL"].abs())*.5
    
    ## time importance VP
    FFNN_wmq_SDF_VI_category = abs(FFNN_wmq_SDF_VI).copy()
    FFNN_wmq_SDF_VI_category["category"] = [*["w"]*41,*["m"]*26,*["q"]*76]
    FFNN_wmq_SDF_VI_category = FFNN_wmq_SDF_VI_category.groupby("category").sum()
    FFNN_wmq_SDF_VI_category.sort_values("abs",ascending=False,inplace=True)
    # del FFNN_wmq_RP_VI_category["abs"]
    
    FFNN_wmq_SDF_VI["category"] = [*["w"]*41,*["m"]*26,*["q"]*76]
    FFNN_wmq_SDF_VI.sort_values("abs",ascending=False,inplace=True)
    plot_VI(FFNN_wmq_SDF_VI, mode="single",display = 10,objective="SDF", 
            name=["Top 10 wmq SDF model"])
    

class color_shader_256():
    def __init__(self,base_color:tuple = (255/256,0,0),
                 target_color:tuple= (255/256,200/256,200/256),n_steps = 5):
        self.base_color = base_color
        self.target_color = target_color
        self.n_steps = n_steps
        self.reset()
    def reset(self):
        self.color = self.base_color
        self.step_n = -1
    def __getitem__(self,step_n=None):
        self.step_n += 1
        if step_n is not None:
            self.step_n = step_n
        color = (self.base_color[0]+(self.target_color[0]-self.base_color[0])*\
                     self.step_n/(self.n_steps-1),
                 self.base_color[1]+(self.target_color[1]-self.base_color[1])*\
                     self.step_n/(self.n_steps-1),
                 self.base_color[2]+(self.target_color[2]-self.base_color[2])*\
                     self.step_n/(self.n_steps-1),
                     )
        return color

def sigmoid_kernel(value,weight,bias):
    return 1-1/(1+np.exp(-value*weight*8+weight*4-bias))
    
def get_sigmoid_decay(weight, bias):
    # weight = 0.0000000000612; bias = 0.0000000000544;
    series = np.linspace(0, 1.05, 1000)
    y_values = sigmoid_kernel(series,weight,bias)
    
    return y_values, series

def plot_sigmoid_basis():
    fontsize = 15
    standard_weights = [1,1,1,.3,0,-1]
    standard_biases = [0,1,-1,+1,0,0]
    labels = ["Initialised","Shift to left","Shift to right","stretched and shifted","flat","reversed"]
    colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a6cee3']
    plt.figure(figsize=(10, 6))  # Set the figure size
   
    for weight, bias, label, color in zip(standard_weights, standard_biases, labels, colors):
        y_values, series = get_sigmoid_decay(weight, bias)
        plt.plot(series, y_values, label=label, linewidth = 3, color = color)  # Plot each curve with a label
 
    # Customize the plot
    plt.xlabel('Input Value',fontsize = fontsize)
    plt.ylabel('Sigmoid Kernel Output',fontsize = fontsize)
    plt.xlim(0, 1)  # Set x-axis limits
    plt.ylim(0, 1)  # Set y-axis limits
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title('Sigmoid Kernel with Different Parameter Configurations',fontsize = fontsize)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,fontsize = fontsize)
    plt.grid(True)
    plt.savefig(PROCESSED_PATH+"raw_reversed_sigmoid.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_sigmoid_decay(data:dict, y_max:int = 31, name:str= ""):
    """
    Generates plots of sigmoid decay function

    Parameters
    ----------
    data : dict
        Dictionary of two-element lists that contain the parameters for the sigmoid layer.
    y_max : int, DEFAULT 31
        Number of days to rescale the graphs y-axis to.
    name : str, DEFAULT ""
        Name for the graph in the files.

    Returns
    -------
    None.


    Testing parameters:
    -------------------
        data    = {i: parameters[i]["m"] for i in parameters.keys()}
        y_max   = 91;
        name    = "SDF month kernel"
    """
    
    # 
    
    years = list(data.keys())
    n_steps = len(data[years[0]])
    years.sort()
    # year_difference = years[-1]-years[0] 
    # year_color_inrement = [
    #     color_shader_256((1,0,0),(1,.5,.5),n_steps=len(years)),
    #     color_shader_256((0,.25,.5),(0,0,1),n_steps=len(years))]
    year_color_inrement = [
        color_shader_256((1,0,1),(1,0,1),n_steps=len(years)),
        color_shader_256((.2,.8,.2),(.2,.8,.2),n_steps=len(years))]
    year_color_index = 0
    
    ## create 3d plot
    w = 12; h = 6;
    fig = plt.figure(figsize=(w,h))#, constrained_layout=True) 
    ax = plt.gca()
    # ax.set_axis_off()
    ax.xaxis.label.set_color('white')        #setting up X-axis label color to yellow
    ax.yaxis.label.set_color('white')          #setting up Y-axis label color to blue
    
    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black
    
    for side in ["left", "top","bottom","right"]:
        ax.spines[side].set_color('white')        # setting up side of plot tick color to white
   
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    ax.figure.set_size_inches(float(w)/(r-l), float(h)/(t-b))
    
        
    ax = fig.add_subplot(projection='3d')
    fig.set_constrained_layout_pads(w_pad=2.0, h_pad=2.0, hspace=0.2, wspace=0.2)
    
    # y_plane = 0
    # x_plane = np.linspace(2003, 2022, 500)   # Cover x-axis range
    # z_plane = np.linspace(0, 1, 500)        # Cover z-axis range
    # # Create a meshgrid for the x and z ranges
    # X, Z = np.meshgrid(x_plane, z_plane)
    # # Plot the y=0 plane as a semi-transparent surface
    # ax.plot_surface(X, np.full_like(X, y_plane), Z, color='gray', alpha=0.6)
    
    ## 1 by 1 add plots
    for year in years:
        year_color = year_color_inrement[year_color_index%2][year-years[0]]
        year_color_index+=1
        
        year_increment = 1/(4+n_steps-1)
        increment = year_increment*2
        for weight, bias in data[year]:
            y_data, x_data = get_sigmoid_decay(weight, bias)
            # year_values = np.array([year + increment]*len(x_data))
            year_values = np.array([year]*len(x_data))
            ax.plot(year_values,x_data,y_data,# ls = "--",
                    color = year_color, linewidth = 1, alpha=1)
            increment += year_increment 
    
    
    ax.set_xlabel('Year', labelpad=25)
    ax.set_ylabel('Date Difference')
    ax.set_zlabel('Reverese Sigmoid Decay')
    # plt.rcParams['axes.titlepad'] = -20
    plt.rcParams['axes.titley'] = .8
    plt.title("Reversed Sigmoid Kernel: "+name)
    plt.xlim(years[0]-.25,years[-1]+.25)
    plt.ylim(0,1)
    ax.set_box_aspect([4, 1, 1]) #, zoom=0.9)
    ax.view_init(elev=20, azim=60) 
    ax.set_zlim(0,1)
    
        ## x_tick locations
    tick_locations = range(2003,2023,2)
    tick_labels = [str(i) for i in tick_locations]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels)
    # plt.legend()
   
    ## ytick locations 
    tick_labels = [int(i*y_max/3) for i in range(4)]
    tick_locations = [i/y_max for i in tick_labels]
    ax.set_yticks(tick_locations)
    ax.set_yticklabels(tick_labels)
    plt.subplots_adjust(left=0.4)
   
    plt.grid(True)
    fig.tight_layout()
    ax.zaxis.labelpad = 15
    # plt.subplots_adjust(left=0.5)
    # fig.tight_layout(rect=[1,1, 2, 2])
    # plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
    plt.show()
    # plt.rcParams['axes.titlepad'] = 0
    plt.rcParams['axes.titley'] = None
    fig.savefig(PROCESSED_PATH+"TimeDecay "+name+".pdf", dpi=200, bbox_inches = "tight")
               
    

def time_decay_import(model_name = "FFNN_wmq_VP", num_models=5, years= YEARS):
    '''
    Returns the time_decay parameters for a given model

    Parameters
    ----------
    model_name : TYPE, optional
        DESCRIPTION. The default is "FFNN_w_VP".
    num_models : TYPE, optional
        DESCRIPTION. The default is 5.
    years : TYPE, optional
        DESCRIPTION. The default is YEARS.

    Returns
    -------
    None.


    Testing Parameters:
    -------------------
        model_name  = "FFNN_wmq_VP"
        num_models  = 5
        years       = YEARS
        
        model_name  = "FFNN_wmq_RP"
        
        model_name  = "FFNN_wmq_SDF"
    '''
    parameters = {"m":{},"q":{}}
    
    for year in years:
        # year = 2003
        parameters["m"][year] = []; parameters["q"][year] = [];
        print(f"\n{model_name} {year}:")
        for i_model in range(num_models):
            # i_model = 1
            parameters_local = to.load(MODELS_PATH+f"{model_name}_{i_model}_{year}.pkl")
            w_m = float(parameters_local['kernels_postprocessor.0.weights'].cpu().detach())
            b_m = float(parameters_local['kernels_postprocessor.0.bias'].cpu().detach())
            w_q = float(parameters_local['kernels_postprocessor.1.weights'].cpu().detach())
            b_q = float(parameters_local['kernels_postprocessor.1.bias'].cpu().detach())
            
            print(
                f"\t{i_model}  weight m", f"{w_m: 15.13f}", "bias m", f"{b_m: 15.13f}",
                "weight q", f"{w_q: 15.13f}", "bias q", f"{b_q: 15.13f}")
            
            parameters["m"][year].append([w_m,b_m])
            parameters["q"][year].append([w_q,b_q])
    plot_sigmoid_decay(parameters["m"],31,model_name[9:]+" month kernel")
    plot_sigmoid_decay(parameters["q"],91,model_name[9:]+" quarter kernel")
    
    

if False:
    pass    
    

################################################################################
################################################################################
################################################################################
###########################   Portfolio Performance   ##########################
################################################################################
################################################################################
################################################################################

if False:
    ## create tables and graphs for realised volatility prediction:
        
    
    pass




