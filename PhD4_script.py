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

- replace PATH_base

This script is organised in stages, each responsible for a specific part of the results.
These stages are:
    
    Stage1 : 'benchmarks'     , Calculation of all benchmark models 
    
    Stage2 : 'hyperparameters', Optimisation of hyperparameters for all neural networks
    
    Stage3 : 'Neural Networks', Training of neural networks
    
The strings inside the quotation marks '' after each stage above lead to 
    execution of the respective code below, when put in the list executions.
    

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
executions = [] # contains one or more of "benchmarks", "hyperparameters", "Neural Networks"
phd3_dataset_variants = ["f500","complete"]
variant = "f500" # "complete"
test = False

import os, pandas as pd, datetime as dt, statsmodels.api as sm, threading as th,\
    numpy as np, torch as to, copy, json
from torch import nn
from backend_functions import device, model, PhD4_dataset, print_hint2, PPB_norm_df,\
    RP_objective, SDF_objective, Loss_ENET, Loss_SDF, PPB_rank_df,PPB_nan_to_0,\
    Loss_MSE_weighted, isnumber, PPB_nan_mean_df, Loss_Huber_weighted, FFNN,\
    Layers_multifreq, Scheduler_Monitor, BatchNorm_self, OBJ_P_weight_power,\
    OBJ_P_weights_stdmin, Layers_FFNN, PPB_nan_drop_df, Layer_Sigmoid_decay,\
    OBJ_P_long_short_ratio, ANN_MultiFreq, act_DoubleTanh, Loss_QLikelihood, \
    PhD2_year_job_thread, fc_naive, Layers_TNN_CNN_FFNN, Layers_FFNN_CNN, CNN_TNN
# from backend_functions import * 


## mode of data storage
global_data_mode = "numpy"
if device != "cpu":
    global_data_mode = "torch"

FILE_PATH = "D:/Backup Research/"
if not os.path.exists(FILE_PATH):
    FILE_PATH = "C:/BT/"

PATH_base = FILE_PATH + "Data_P3/"
DATA_PATH = FILE_PATH + f"Data_P3/{variant}/"
HYPERPARAMETER_PATH = FILE_PATH+"Paper3 results/Hyperparameters/" 
RESULTS_PATH = FILE_PATH +"Paper3 results/Iterations/"
MODELS_PATH = FILE_PATH + "Paper3 results/Models/"
PROCESSED_PATH = FILE_PATH + "Paper3 results/Processed/"
TIMER_PATH = FILE_PATH +"Paper3 results/Timers/"

TARGET_VARIABLES ={
    "SDF":"ret1w",
    "RP":"ret1w",
    "VP":"rvarf1w",
    "VP_HARQ":"rvarf1w"}

HARQ_columns = ["rvar1w", "rvar4w", "rvar13w", "rq1w", "rq4w", "rq13w"]

DROP_NA_columns =  [
    "b", "b2", "bas", "im1w", "im26w", "imcap", "irv26w", "irv156w", "mcap", "mdr1w", 
    "mdr4w", "sto1w", "tv1w", "vst1w", "vtv1w", "sto4w", "tv4w", "vst4w", "vtv4w", 
    "m1w", "m4w", "m13w", "m26w", "rvar1w", "rvar4w", "rvar13w", "rvar26w", "irvar13w", 
    "itv1w", "itv4w", "imdr4w", "ib", "iirv26w", "rq1w", "rq4w", "rq13w"]

if test==True:
    ## global default parameters
    N_PARALLEL = 2
    N_MODELS = 2
    test_appendix = "-test"
else:
    N_PARALLEL = 5
    N_MODELS = 5
    test_appendix = ""
    
years_execution = list(range(2003,2023))
    
####################################################################################################
####################################################################################################
####################################################################################################
##################################   Benchmark models (Stage 1)   ##################################
####################################################################################################
####################################################################################################
####################################################################################################

################################################################################
################################################################################
######################  Volatility Prediction Benchmarks  ######################
################################################################################
################################################################################

if False:
    ## date range:
    years = list(range(2003,2023))
    
    ## target variable
    target_variable = "rvarf1w"
    date_entity = "date"
    
    
    HARQF_model_columns = ["const","rvar1w","rvar4w","rvar13w","rq1w","rq4w","rq13w"]
    HAR_model_columns = ["const","rvar1w","rvar4w","rvar13w"]
    HARQ_model_columns = ["const","rvar1w","rvar4w","rvar13w","rq1w"]
    AR1_model_columns = ["const","rvar1w"]
    AR5_model_columns = ["const","rvar1w",*[f"rvar1wt-{lag:d}" for lag in range(1,5)]]
    
    
    
    ## empty results container
    predictions = { 
        "test":[],"test_KS":[],
        "AR1":[],"AR5":[],"HAR":[],"HARQ":[],"HARQ-F":[],"KS":[]}
    prediction_stats = {
        "AR1":[],"AR5":[],"HAR":[],"HARQ":[],"HARQ-F":[],"KS":[]}
    prediction_results = {
        "AR1":[],"AR5":[],"HAR":[],"HARQ":[],"HARQ-F":[],"KS":[]}

    QLike = Loss_QLikelihood(mode = "cap")
    
    for year in years_execution:
        # year = 2003
        input_data = pd.read_hdf(PATH_base+ f"{variant}_benchmark/HARx_{variant:s}.h5",key="data").sort_index()
        input_data_ks = pd.read_hdf(PATH_base+ f"{variant}_benchmark/ALLWx_{variant:s}.h5",key="data").sort_index()
        output_data = pd.read_hdf(PATH_base+ f"{variant}_benchmark/HARy_{variant:s}.h5",key="data")[target_variable].sort_index()
        
        for lag in range(1,5):
            input_data[f"rvar1wt-{lag:d}"] = input_data["rvar1w"].groupby("gvkey").shift(lag)
        
        train_start = dt.datetime.strptime(f"{year-11}-12-31","%Y-%m-%d").date()
        train_end = dt.datetime.strptime(f"{year-1}-12-31","%Y-%m-%d").date() 
        test_end = dt.datetime.strptime(f"{year}-12-31","%Y-%m-%d").date()
        
        ## ensure no NA
        index_mask = ~(output_data.isna()) #(input_data.isna().sum(axis=1)==0)&(~(output_data.isna()))
        
        input_data = input_data.replace(np.nan,0)
        
        ## train data
        local_train_input = sm.add_constant(input_data[
            (input_data.index.get_level_values("date")<=train_end)&\
            (input_data.index.get_level_values("date")>train_start)& index_mask])
        
        local_train_output = output_data[
            (output_data.index.get_level_values("date")<=train_end)&\
            (output_data.index.get_level_values("date")>train_start)& index_mask]
    
        ## test data
        local_test_input = sm.add_constant(input_data[
            (input_data.index.get_level_values("date")<=test_end)&\
            (input_data.index.get_level_values("date")>train_end)& index_mask])
        
        local_test_output = output_data[
            (output_data.index.get_level_values("date")<=test_end)&\
            (output_data.index.get_level_values("date")>train_end)& index_mask]
            
        ### HAR model
        
        print_hint2(f"Computing HAR model for {variant:s} dataset")
        print("train_start: ",train_start, "\ntrain_end  : ",train_end, "\ntest_end   : ",test_end)
        HAR_model = sm.OLS(local_train_output,local_train_input[HAR_model_columns]).fit()
    
        print(HAR_model.summary())
        
        pred_is     = HAR_model.predict(local_train_input[HAR_model_columns])
        pred_oos    = HAR_model.predict(local_test_input[HAR_model_columns])
        
        scores = {
            "MSE_is": ((pred_is-local_train_output)**2).mean(),
            "MSE_oos":((pred_oos-local_test_output)**2).mean(),
            "R2_is": 1-((pred_is-local_train_output)**2).sum() / ((local_train_output-local_train_output.mean())**2).sum(),
            "R2_oos":1-((pred_oos-local_test_output)**2).sum() / ((local_test_output-local_test_output.mean())**2).sum(),
            "R2_modified_is": 1-((pred_is-local_train_output)**2).sum() / (local_train_output**2).sum(),
            "R2_modified_oos":1-((pred_oos-local_test_output)**2).sum() / (local_test_output**2).sum(),
            "QLike_is":QLike(pred_is,local_train_output),
            "QLike_oos":QLike(pred_oos,local_test_output),
            "Model":"HAR", "year": f"{year}-12-31"}
        print(pd.DataFrame([scores]).T,end = "\n\n")
        
        predictions["test"].append(local_test_output)
        predictions["HAR"].append(pred_oos)
        prediction_results["HAR"].append(scores)
        prediction_stats["HAR"].append(pd.DataFrame({
            "avg_pred":pred_oos.groupby(date_entity).mean(),
            # "l_ratio":(pred_oos!=0).sum()/len(pred_oos),
            # "s_ratio":(pred_oos==0).sum()/len(pred_oos),
            "std_pred":pred_oos.groupby(date_entity).std(),
            "min_pred":pred_oos.groupby(date_entity).min(),
            "max_pred":pred_oos.groupby(date_entity).max(),
            "skewness":pred_oos.groupby(date_entity).skew(),
            "kurtorsis":pred_oos.groupby(date_entity).apply(pd.Series.kurt)}))
        
        
        ### HARQ model
        
        ## HARQ
        
        print_hint2(f"Computing HAR model for {variant:s} dataset")
        print("train_start: ",train_start, "\ntrain_end  : ",train_end, "\ntest_end   : ",test_end)
        HARQ_model = sm.OLS(local_train_output,local_train_input[HARQ_model_columns]).fit()
    
        print(HARQ_model.summary())
        
        pred_is     = HARQ_model.predict(local_train_input[HARQ_model_columns])
        pred_oos    = HARQ_model.predict(local_test_input[HARQ_model_columns])
        
        scores = {
            "MSE_is": ((pred_is-local_train_output)**2).mean(),
            "MSE_oos":((pred_oos-local_test_output)**2).mean(),
            "R2_is": 1-((pred_is-local_train_output)**2).sum() / ((local_train_output-local_train_output.mean())**2).sum(),
            "R2_oos":1-((pred_oos-local_test_output)**2).sum() / ((local_test_output-local_test_output.mean())**2).sum(),
            "R2_modified_is": 1-((pred_is-local_train_output)**2).sum() / (local_train_output**2).sum(),
            "R2_modified_oos":1-((pred_oos-local_test_output)**2).sum() / (local_test_output**2).sum(),
            "QLike_is":QLike(pred_is,local_train_output),
            "QLike_oos":QLike(pred_oos,local_test_output),
            "Model":"HARQ", "year": f"{year}-12-31"
            }
        print(pd.DataFrame([scores]).T,end = "\n\n")
        
        predictions["HARQ"].append(pred_oos)
        prediction_results["HARQ"].append(scores)
        prediction_stats["HARQ"].append(pd.DataFrame({
            "avg_pred":pred_oos.groupby(date_entity).mean(),
            "l_ratio":(pred_oos!=0).sum()/len(pred_oos),
            "s_ratio":(pred_oos==0).sum()/len(pred_oos),
            "std_pred":pred_oos.groupby(date_entity).std(),
            "min_pred":pred_oos.groupby(date_entity).min(),
            "max_pred":pred_oos.groupby(date_entity).max(),
            "skewness":pred_oos.groupby(date_entity).skew(),
            "kurtorsis":pred_oos.groupby(date_entity).apply(pd.Series.kurt)}))
        
        '''
        HARQ
        R2 in-sample     : 0.24351607096143746
        R2 out-of-sample : 0.29786819436485057
        '''
        
        ## HARQ-F
        
        print_hint2(f"Computing HAR model for {variant:s} dataset")
        print("train_start: ",train_start, "\ntrain_end  : ",train_end, "\ntest_end   : ",test_end)
        HARQF_model = sm.OLS(local_train_output,local_train_input[HARQF_model_columns]).fit()
    
        print(HARQF_model.summary())
        
        pred_is     = HARQF_model.predict(local_train_input[HARQF_model_columns])
        pred_oos    = HARQF_model.predict(local_test_input[HARQF_model_columns])
        
        scores = {
            "MSE_is": ((pred_is-local_train_output)**2).mean(),
            "MSE_oos":((pred_oos-local_test_output)**2).mean(),
            "R2_is": 1-((pred_is-local_train_output)**2).sum() / ((local_train_output-local_train_output.mean())**2).sum(),
            "R2_oos":1-((pred_oos-local_test_output)**2).sum() / ((local_test_output-local_test_output.mean())**2).sum(),
            "R2_modified_is": 1-((pred_is-local_train_output)**2).sum() / (local_train_output**2).sum(),
            "R2_modified_oos":1-((pred_oos-local_test_output)**2).sum() / (local_test_output**2).sum(),
            "QLike_is":QLike(pred_is,local_train_output),
            "QLike_oos":QLike(pred_oos,local_test_output),
            "Model":"HARQ-F", "year": f"{year}-12-31"
            }
        print(pd.DataFrame([scores]).T,end = "\n\n")
        
        predictions["HARQ-F"].append(pred_oos)
        prediction_results["HARQ-F"].append(scores)
        prediction_stats["HARQ-F"].append(pd.DataFrame({
            "avg_pred":pred_oos.groupby(date_entity).mean(),
            "l_ratio":(pred_oos!=0).sum()/len(pred_oos),
            "s_ratio":(pred_oos==0).sum()/len(pred_oos),
            "std_pred":pred_oos.groupby(date_entity).std(),
            "min_pred":pred_oos.groupby(date_entity).min(),
            "max_pred":pred_oos.groupby(date_entity).max(),
            "skewness":pred_oos.groupby(date_entity).skew(),
            "kurtorsis":pred_oos.groupby(date_entity).apply(pd.Series.kurt)}))
        
        '''
        HARQ-F
        R2 in-sample     : 0.25472597123341056
        R2 out-of-sample : 0.30434374770150974
        '''
        
        ### SHAR
        ''' 
        Performs slightly worse than HARQ usually.
        Left out for reasons of simplicity, measures would need to be caluclated 
            additionally based on daily data.
        Could potentially be taken into account.
        '''
        
        ### AR 
        ## AR 1
        print_hint2(f"Computing AR model for {variant:s} dataset")
        print("train_start: ",train_start, "\ntrain_end  : ",train_end, "\ntest_end   : ",test_end)
        AR1_model = sm.OLS(local_train_output,local_train_input[AR1_model_columns]).fit()
    
        print(AR1_model.summary())
        
        pred_is     = AR1_model.predict(local_train_input[AR1_model_columns])
        pred_oos    = AR1_model.predict(local_test_input[AR1_model_columns])
        
        scores = {
            "MSE_is": ((pred_is-local_train_output)**2).mean(),
            "MSE_oos":((pred_oos-local_test_output)**2).mean(),
            "R2_is": 1-((pred_is-local_train_output)**2).sum() / ((local_train_output-local_train_output.mean())**2).sum(),
            "R2_oos":1-((pred_oos-local_test_output)**2).sum() / ((local_test_output-local_test_output.mean())**2).sum(),
            "R2_modified_is": 1-((pred_is-local_train_output)**2).sum() / (local_train_output**2).sum(),
            "R2_modified_oos":1-((pred_oos-local_test_output)**2).sum() / (local_test_output**2).sum(),
            "QLike_is":QLike(pred_is,local_train_output),
            "QLike_oos":QLike(pred_oos,local_test_output),
            "Model":"AR1", "year": f"{year}-12-31"
            }
        print(pd.DataFrame([scores]).T,end = "\n\n")
        
        predictions["AR1"].append(pred_oos)
        prediction_results["AR1"].append(scores)
        prediction_stats["AR1"].append(pd.DataFrame({
            "avg_pred":pred_oos.groupby(date_entity).mean(),
            "l_ratio":(pred_oos!=0).sum()/len(pred_oos),
            "s_ratio":(pred_oos==0).sum()/len(pred_oos),
            "std_pred":pred_oos.groupby(date_entity).std(),
            "min_pred":pred_oos.groupby(date_entity).min(),
            "max_pred":pred_oos.groupby(date_entity).max(),
            "skewness":pred_oos.groupby(date_entity).skew(),
            "kurtorsis":pred_oos.groupby(date_entity).apply(pd.Series.kurt)}))
        
        '''
        AR 5:
            R2 in-sample     : 0.14786638186529732
            R2 out-of-sample : 0.12643025462298085
        '''
        
        ### AR 
        ## AR 5
        print_hint2(f"Computing AR model for {variant:s} dataset")
        print("train_start: ",train_start, "\ntrain_end  : ",train_end, "\ntest_end   : ",test_end)
        AR5_model = sm.OLS(local_train_output,local_train_input[AR5_model_columns]).fit()
    
        print(AR5_model.summary())
        
        pred_is     = AR5_model.predict(local_train_input[AR5_model_columns])
        pred_oos    = AR5_model.predict(local_test_input[AR5_model_columns])
        
        scores = {
            "MSE_is": ((pred_is-local_train_output)**2).mean(),
            "MSE_oos":((pred_oos-local_test_output)**2).mean(),
            "R2_is": 1-((pred_is-local_train_output)**2).sum() / ((local_train_output-local_train_output.mean())**2).sum(),
            "R2_oos":1-((pred_oos-local_test_output)**2).sum() / ((local_test_output-local_test_output.mean())**2).sum(),
            "R2_modified_is": 1-((pred_is-local_train_output)**2).sum() / (local_train_output**2).sum(),
            "R2_modified_oos":1-((pred_oos-local_test_output)**2).sum() / (local_test_output**2).sum(),
            "QLike_is":QLike(pred_is,local_train_output),
            "QLike_oos":QLike(pred_oos,local_test_output),
            "Model":"AR5", "year": f"{year}-12-31"
            }
        print(pd.DataFrame([scores]).T,end = "\n\n")
        
        predictions["AR5"].append(pred_oos)
        prediction_results["AR5"].append(scores)
        prediction_stats["AR5"].append(pd.DataFrame({
            "avg_pred":pred_oos.groupby(date_entity).mean(),
            "l_ratio":(pred_oos!=0).sum()/len(pred_oos),
            "s_ratio":(pred_oos==0).sum()/len(pred_oos),
            "std_pred":pred_oos.groupby(date_entity).std(),
            "min_pred":pred_oos.groupby(date_entity).min(),
            "max_pred":pred_oos.groupby(date_entity).max(),
            "skewness":pred_oos.groupby(date_entity).skew(),
            "kurtorsis":pred_oos.groupby(date_entity).apply(pd.Series.kurt)}))
        
        '''
        AR 5:
            R2 in-sample     : 0.14786638186529732
            R2 out-of-sample : 0.12643025462298085
        '''
    
        ### Kitchen sink model
        
        normaliser = PPB_norm_df(); nan_handler = PPB_nan_to_0()
        use_columns = ["b", "b2", "bas", "im1w", "im26w", "imcap", "irv26w", "irv156w", 
                       "mcap", "mdr1w", "mdr4w", "sto1w", "tv1w", "vst1w", "vtv1w", 
                       "sto4w", "tv4w", "vst4w", "vtv4w", "m1w", "m4w", "m13w", "m26w", 
                       "rvar1w", "rvar4w", "rvar13w", "rvar26w", "irvar13w", "itv1w", 
                       "itv4w", "imdr4w", "ib", "iirv26w", "rq1w", "rq4w", "rq13w"]
        input_data_ks = input_data_ks[use_columns]
        input_data_ks = normaliser(input_data_ks)
        input_data_ks = nan_handler(input_data_ks) 
        output_data = output_data.loc[ output_data.index.isin(input_data_ks.index),:]
        input_data_ks.sort_index(level = ["gvkey","date"],inplace=True)
        output_data.sort_index(level = ["gvkey","date"],inplace=True)
        
        # input_data_ks = input_data_ks.replace(np.nan,0)
        # output_data = output_data.replace(np.nan,0)
        ## ensure no NA
        index_mask_ks = (input_data_ks.isna().sum(axis=1)==0)&\
            (~(output_data.isna()))
            
        local_train_input_ks = sm.add_constant(input_data_ks[
            (input_data_ks.index.get_level_values("date")<=train_end)&\
            (input_data_ks.index.get_level_values("date")>train_start)& index_mask_ks])
            
        local_train_output_ks = output_data[
             (output_data.index.get_level_values("date")<=train_end)&\
             (output_data.index.get_level_values("date")>train_start)& index_mask_ks]
        ## test data
        local_test_input_ks = sm.add_constant(input_data_ks[
            (input_data_ks.index.get_level_values("date")<=test_end)&\
            (input_data_ks.index.get_level_values("date")>train_end)& index_mask_ks])
    
        local_test_output_ks = output_data[
            (output_data.index.get_level_values("date")<=test_end)&\
            (output_data.index.get_level_values("date")>train_end)& index_mask_ks]
    
    
        print_hint2(f"Computing kitchen sink model for {variant:s} dataset")
        print("train_start: ",train_start, "\ntrain_end  : ",train_end, "\ntest_end   : ",test_end)
        KS_model = sm.OLS(local_train_output_ks,local_train_input_ks).fit()
    
        print(KS_model.summary())
        
        pred_is     = KS_model.predict(local_train_input_ks)
        pred_oos    = KS_model.predict(local_test_input_ks)
        
        scores = {
            "MSE_is": ((pred_is-local_train_output_ks)**2).mean(),
            "MSE_oos":((pred_oos-local_test_output_ks)**2).mean(),
            "R2_is": 1-((pred_is-local_train_output_ks)**2).sum() / ((local_train_output_ks-local_train_output_ks.mean())**2).sum(),
            "R2_oos":1-((pred_oos-local_test_output_ks)**2).sum() / ((local_test_output_ks-local_test_output_ks.mean())**2).sum(),
            "R2_modified_is": 1-((pred_is-local_train_output_ks)**2).sum() / (local_train_output_ks**2).sum(),
            "R2_modified_oos":1-((pred_oos-local_test_output_ks)**2).sum() / (local_test_output_ks**2).sum(),
            "QLike_is":QLike(pred_is,local_train_output_ks),
            "QLike_oos":QLike(pred_oos,local_test_output_ks),
            "Model":"KS", "year": f"{year}-12-31"
            }
        print(pd.DataFrame([scores]).T,end = "\n\n")
        
        predictions["test_KS"].append(local_test_output_ks)
        predictions["KS"].append(pred_oos)
        prediction_results["KS"].append(scores)
        prediction_stats["KS"].append(pd.DataFrame({
            "avg_pred":pred_oos.groupby(date_entity).mean(),
            "l_ratio":(pred_oos!=0).sum()/len(pred_oos),
            "s_ratio":(pred_oos==0).sum()/len(pred_oos),
            "std_pred":pred_oos.groupby(date_entity).std(),
            "min_pred":pred_oos.groupby(date_entity).min(),
            "max_pred":pred_oos.groupby(date_entity).max(),
            "skewness":pred_oos.groupby(date_entity).skew(),
            "kurtorsis":pred_oos.groupby(date_entity).apply(pd.Series.kurt)}))
        '''
        Kitchen sink
        
        R2 in-sample     : 0.11035032501861297
        R2 out-of-sample : -0.48139464180852665
        '''
    for key in predictions.keys():
        predictions[key] = pd.concat(predictions[key])
    predictions["TS"] = pd.concat([
        predictions["test"].to_frame("test"), predictions["AR1"].to_frame("AR1"), 
        predictions["AR5"].to_frame("AR5"), predictions["HAR"].to_frame("HAR"), 
        predictions["HARQ"].to_frame("HARQ"), predictions["HARQ-F"].to_frame("HARQ-F")],axis=1)
    predictions["KS"] = pd.concat([
        predictions["test_KS"].to_frame("test_KS"), predictions["KS"].to_frame("KS")],axis=1)
    predictions = {x:predictions[x] for x in ["TS","KS"]}
    
    mcap = pd.read_hdf(PATH_base+f"{variant}_benchmark/mcap.h5",key="data")
    predictions["TS"] = pd.merge(predictions["TS"], mcap, how = "inner",
                                 left_index=True, right_index=True)
    
    predictions["KS"] = pd.merge(predictions["KS"], mcap, how = "inner",
                                 left_index=True, right_index=True)
    predictions["TS"].to_hdf(RESULTS_PATH+ f"Benchmarks/VP_TS_pred.h5","data")
    predictions["KS"].to_hdf(RESULTS_PATH+ f"Benchmarks/VP_KS_pred.h5","data")
    
    predictions_dec1 = {}
    predictions_dec10 = {}
    
    for table in ["TS","KS"]:
        # table = "KS"
        deciles = predictions[table]["mcap"].groupby("date").quantile(.9).to_frame(name="dec10")
        deciles["dec1"] = predictions[table]["mcap"].groupby("date").quantile(.1)
        predictions[table] = pd.merge(predictions[table],deciles,left_index=True, right_index=True)
        predictions_dec1[table] = predictions[table][predictions[table]["mcap"]<predictions[table]["dec1"]]
        predictions_dec10[table] = predictions[table][predictions[table]["mcap"]>predictions[table]["dec10"]]
    
    for key in prediction_results.keys():
        # key = "HARQ"
        prediction_results[key] = pd.DataFrame(prediction_results[key])
        prediction_results[key].set_index("year",inplace=True)
    
    prediction_results_total = {}
    
    qlike = Loss_QLikelihood()
    for column in ["AR1","AR5","HAR","HARQ","HARQ-F", "KS"]:
        # column = "AR1"
        table = "TS"; test_column = "test"
        if column == "KS":
            table = "KS"; test_column = "test_KS"
        
        prediction_results_total[column] = {}
        
        ## MSE
        prediction_results_total[column]["MSE"] = {}
        prediction_results_total[column]["MSE"]["Sample"] = ((
            predictions[table][test_column]-predictions[table][column])**2).mean()
        
        prediction_results_total[column]["MSE"]["Small"]  = ((
            predictions_dec1[table][test_column]-predictions_dec1[table][column])**2).mean()
        prediction_results_total[column]["MSE"]["Large"]  = ((
            predictions_dec10[table][test_column]-predictions_dec10[table][column])**2).mean()
        
        ## QLikelihood
        
        prediction_results_total[column]["Q-like"] = {}
        prediction_results_total[column]["Q-like"]["Sample"] = qlike(
            predictions[table][column],predictions[table][test_column])
        prediction_results_total[column]["Q-like"]["Small"] = qlike(
            predictions_dec1[table][column],predictions_dec1[table][test_column])
        prediction_results_total[column]["Q-like"]["Large"] = qlike(
            predictions_dec10[table][column],predictions_dec10[table][test_column],)
            
            
            
        ## R2-pred
        prediction_results_total[column]["R2"] = {}
        prediction_results_total[column]["R2"]["Sample"] = 1-((
            predictions[table][test_column]-predictions[table][column])**2).sum()/((
                predictions[table][test_column]-predictions[table][test_column].mean())**2).sum()
        
        prediction_results_total[column]["R2"]["Small"]  = 1-((
            predictions_dec1[table][test_column]-predictions_dec1[table][column])**2).sum()/((
                predictions_dec1[table][test_column]-predictions_dec1[table][test_column].mean())**2).sum()
        prediction_results_total[column]["R2"]["Large"]  = 1-((
            predictions_dec10[table][test_column]-predictions_dec10[table][column])**2).sum()/((
                predictions_dec10[table][test_column]-predictions_dec10[table][test_column].mean())**2).sum()
        
        
        ## R2-pred modified
        prediction_results_total[column]["R2_modified"] = {}
        prediction_results_total[column]["R2_modified"]["Sample"] = 1-((
            predictions[table][test_column]-predictions[table][column])**2).sum()/(predictions[table][test_column]**2).sum()
        
        prediction_results_total[column]["R2_modified"]["Small"]  = 1-((
            predictions_dec1[table][test_column]-predictions_dec1[table][column])**2).sum()/(predictions_dec1[table][test_column]**2).sum()
        prediction_results_total[column]["R2_modified"]["Large"]  = 1-((
            predictions_dec10[table][test_column]-predictions_dec10[table][column])**2).sum()/(predictions_dec10[table][test_column]**2).sum()        
    
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
    reshaped_df.to_csv(PROCESSED_PATH+"VP_benchmark_perf_pred.csv",sep=",")
    
    
    
    ### Analyses over time. 
    monthly_prediction_results = {"R2":{},"MSE":{},"R2_modified":{},"Q-like":{}} # {} # 
    annual_prediction_results = {"R2":{},"MSE":{},"R2_modified":{},"Q-like":{}} # {} # 
    monthly_aggregate = pd.DataFrame()
    annual_aggregate = pd.DataFrame()
    
    for table in ["KS","TS"]:
        predictions[table]["month"] = predictions[table].index.get_level_values("date")
        predictions[table]["month"] = predictions[table]["month"].apply(lambda x: x.strftime("%Y-%m"))
        predictions[table]["year"] = predictions[table]["month"].str[:4]
    
    for column in ["AR1","AR5","HAR","HARQ","HARQ-F", "KS"]:
        
        ## average monthly prediction vs real value
        
        # column = "AR1"
        table = "TS"; test_column = "test"
        if column == "KS":
            table = "KS"; test_column = "test_KS"
        
        
        if column == "AR1":
            monthly_aggregate["RV"] = predictions[table][[test_column,"month"]].groupby("month").mean()
            annual_aggregate["RV"] = predictions[table][[test_column,"year"]].groupby("year").mean()
        
        monthly_aggregate[column] = predictions[table][[column,"month"]].groupby("month").mean()
        annual_aggregate[column] = predictions[table][[column,"year"]].groupby("year").mean()
        
        ## monthly and annual performance measures
        
        monthly_grouped_predicitons = {x:y for x,y in predictions[table].groupby("month")}
        annual_grouped_predicitons = {x:y for x,y in predictions[table].groupby("year")}
                    
        for measure in ["R2","MSE","Q-like","R2_modified"]:
            monthly_prediction_results[measure][column] = {}
            annual_prediction_results[measure][column] = {}
        
        for month in monthly_grouped_predicitons.keys():
            monthly_prediction_results["R2"][column][month] = 1-((
                monthly_grouped_predicitons[month][test_column]- monthly_grouped_predicitons[month][column])**2).sum()/((
                    monthly_grouped_predicitons[month][test_column]- monthly_grouped_predicitons[month][test_column].mean())**2).sum()
                    
            monthly_prediction_results["R2_modified"][column][month] = 1-((
                monthly_grouped_predicitons[month][test_column]- monthly_grouped_predicitons[month][column])**2).sum()/\
                (monthly_grouped_predicitons[month][test_column]**2).sum()
            
            monthly_prediction_results["Q-like"][column][month] =  qlike(
                monthly_grouped_predicitons[month][column],monthly_grouped_predicitons[month][test_column])
            
            monthly_prediction_results["MSE"][column][month] =  ((
                monthly_grouped_predicitons[month][test_column]-monthly_grouped_predicitons[month][column])**2).mean()
            
            
            monthly_prediction_results[column] = {date:{} for date in monthly_grouped_predicitons.keys()}
            annual_prediction_results[column] = {date:{} for date in annual_grouped_predicitons.keys()}
        
        for year in annual_grouped_predicitons.keys():
            annual_prediction_results["R2"][column][year] = 1-((
                annual_grouped_predicitons[year][test_column]- annual_grouped_predicitons[year][column])**2).sum()/((
                    annual_grouped_predicitons[year][test_column]- annual_grouped_predicitons[year][test_column].mean())**2).sum()
                    
            annual_prediction_results["R2_modified"][column][year] = 1-((
                annual_grouped_predicitons[year][test_column]- annual_grouped_predicitons[year][column])**2).sum()/\
                (annual_grouped_predicitons[year][test_column]**2).sum()
            
            annual_prediction_results["Q-like"][column][year] =  qlike(
                annual_grouped_predicitons[year][column],annual_grouped_predicitons[year][test_column])
            
            annual_prediction_results["MSE"][column][year] =  ((
                annual_grouped_predicitons[year][test_column]-annual_grouped_predicitons[year][column])**2).mean()
    
        del monthly_grouped_predicitons, annual_grouped_predicitons

        monthly_aggregate.to_csv(PROCESSED_PATH+"VP_benchmark_resid_monthly.csv")
        annual_aggregate.to_csv(PROCESSED_PATH+"VP_benchmark_resid_annual.csv")
        
        for measure in ["R2","MSE","Q-like","R2_modified"]:
            # measure = "R2"
            
            pd.DataFrame(monthly_prediction_results[measure]).to_csv(PROCESSED_PATH+f"VP_benchmark_{measure}_monthly.csv")
            pd.DataFrame(annual_prediction_results[measure]).to_csv(PROCESSED_PATH+f"VP_benchmark_{measure}_annual.csv")
            
        

####################################################################################################
####################################################################################################
####################################################################################################
#############################   Hyperparameter Optimisation (Stage 2)   ############################
####################################################################################################
####################################################################################################
####################################################################################################
'''
Hyperparameter pre-optimization for models included in the analysis:
    
    FFNN_joined
    FFNN_w
    FFNN_wmq
    
    FFNN_w_HARQ
'''

'''
List of potential hyperparameters:
    
    Dataset related parameters:
        length of training/validation       CHECK
        share of validation data            CHECK
        learning rate                       CHECK
        validation scheme                   CHECK
        preprocessing (normalisation, ranking over x days, nan handling)    CHECK 
        
        lookback setup  (Only lookback models)
        
    loss function:
        base loss       (Only RP and VP)
        alpha           (Only RP and VP)
        lambda          (Only RP and VP)
        
        penalisations   (Only SDF)
    
    Model structure:
        
        number of layers
        number of neurons
        neuron layout
        normalisation
        activation function
        dropout ratio

'''

exclusions = []

##################################################
###################  Functions  ##################
##################################################

def hyperparameter_test(
        model_params, dataset, repetitions, n_parallel, name, 
        to_epoch = 100):    
    
    results = []
    repetitions = list(range(repetitions))
    workers = []
    for worker in range(n_parallel):
        thread_n = th.Thread(
            target=hyperparameter_test_train_worker,
            kwargs={"thread":worker, "name" : name,
                    "dataset":dataset, "model_params":model_params,
                    "repetitions":repetitions, "results":results, 
                    "to_epoch":to_epoch})
        workers.append(thread_n)
    for w in workers:
        w.daemon =True
        w.start()
    for w in workers:
        w.join()
    print("hyperparameter_test: process complete.")
    to.cuda.empty_cache()
    return results    
        
def hyperparameter_test_train_worker(
        model_params, dataset, repetitions ,name, results, thread, to_epoch):
    while len(repetitions) > 0:
        local_repetition = repetitions.pop()
        local_model_pars = copy.deepcopy(model_params)
        local_model_pars["objective"].thread = thread
        model_local = model(**local_model_pars)
        
        model_local.train_validate(
            data_set=dataset, to_epoch=to_epoch,
            fraction=1, frequency = "w",
            tolerance_relative = True, thread = thread)
               
        model_local.test(dataset,frequency="w")
        
        ## extract scores
        best_epoch      = model_local.best_epoch-1
        train_loss      = model_local.objective.epochs_losses["train"][best_epoch]#.cpu().detach().numpy()
        val_loss        = model_local.objective.epochs_losses["val"][best_epoch]
        train_score     = model_local.objective.epochs_scores["train"][best_epoch]#.cpu().detach().numpy()
        val_score       = model_local.objective.epochs_scores["val"][best_epoch]#.cpu().detach().numpy()
        test_loss       = model_local.objective.epochs_losses["test"]
        test_score      = model_local.objective.epochs_scores["test"]
        results.append({
            "best_epoch":best_epoch+1, "train_loss":train_loss, "val_loss":val_loss,
            "train_score":train_score, "val_score":val_score, "test_loss":test_loss,
            "test_score":test_score, "name":name, "rep":local_repetition})

def number_converter(value):
    if isnumber(value):
        value = float(value)
        if value%1==0:
            return int(value)
    return value

def hyperparameter_parse_name(name):
    name = name.split(";")
    pars_dict = {key:number_converter(value) \
                 for key, value in [item.split(":") for item in name]}
    return pars_dict


    
pars_overwrite_model = {
    objective_str: {} for objective_str in ["SDF","RP","VP","VP_HARQ"]}

params_hyperparameter_dataset = {
    "test_date":"2002-12-31",
    "test_len":60,
    "merge_freq": False,
    "batch_size": 2048,
    "data_mode": global_data_mode,
    "file_path":DATA_PATH,
    "input_frequencies":["w"],
    "mode_validation":"vw",
    "return_mode":"simple",
    "preprocessing":{
        "batchwise":{"w":[PPB_rank_df(rolling=0,span=2), PPB_nan_mean_df(rolling=0)]}, 
        "dataset":{}, "filter":{}},
    "lookback": {"w":[1,1]},
    "identifier":"gvkey"
    }

params_hyperparameter_structure = {
    "input_dim":(41,1),
    "channels":1,
    "FFNN":[{"out_features":64},{"out_features":32}, {"out_features":16}, {"out_features":1}],
    "FFNN_norm":nn.BatchNorm1d,
    "FFNN_sec":[{"activation":nn.Sigmoid(), "dropout":0.02}, 
                {"activation":nn.Sigmoid(), "dropout":0.02}, 
                {"activation":nn.Sigmoid(), "dropout":0.02}]
    }

params_hyperparameter_scheduler_monitor = {
    "decline": "linear", "min_delta_relative": True, "skip_batches": 0,
    "n_epochs": 1, "n_epochs_no_change": 20, "min_delta": .01, "max_declines": 5}

params_hyperparameter_model = {
    "model_type": FFNN, "shuffle_epochs": 5, "epoch_schedule": True,
    "side": [],  "scheduler": to.optim.lr_scheduler.StepLR,
    "scheduler_params": {
        "step_size":1,"gamma":np.exp(-2),
        "last_epoch":- 1, "verbose":False}, 
    "device":device,
    "learning_rate" : np.exp(-4),
    # "structure" : params_hyperparameter_structure,
    "tolerance" : 0.00001,
    "max_iter_no_change" : 3*params_hyperparameter_scheduler_monitor["n_epochs_no_change"]}

if 'hyperparameters' in executions:
    
    ############################################################################
    ############################################################################
    ########################  Hyperparameters: Dataset  ########################
    ############################################################################
    ############################################################################
    to_epoch = 24
    
    for objective_str in ["SDF","RP","VP", "VP_HARQ"]:
        # objective_str = "SDF"
        _RESULTS_ = []
        
        if objective_str not in ["VP_HARQ"]:
            continue
        
        target_variable = TARGET_VARIABLES[objective_str]
            
        if objective_str in ["RP","VP","VP_HARQ"]:
            volatility = True if objective_str != "RP" else False
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": Loss_MSE_weighted(), "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":volatility},
                quantiles = 10, target_is_return=not volatility)
            ascending = False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":[
                        OBJ_P_weight_power(), OBJ_P_weights_stdmin()]})
            ascending = True
        if objective_str == "VP_HARQ":
            params_hyperparameter_structure["input_dim"] = (6,1)
        else:
            params_hyperparameter_structure["input_dim"] = (41,1)
        params_hyperparameter_model["structure"] = Layers_FFNN(params_hyperparameter_structure)
        
        
        candidates_load_len = [120,180,300,360]
        candidates_val_share = [.1,.15,.2]
        
        for load_len in candidates_load_len:
            for val_share in candidates_val_share:
                # val_share = .1; load_len = 120;
                
                print_hint2(f"load_len: {load_len:d}, val_share: {val_share:f}",1,1,width_field = 50)
                
                ## determine length of validation
                val_len = int(val_share*load_len)
                
                dataset_hyperparameter_dataset = PhD4_dataset(
                    **params_hyperparameter_dataset,
                    target_variable = target_variable,
                    load_len = load_len,
                    val_len = val_len,
                    use_columns= [] if objective_str != "VP_HARQ" else HARQ_columns
                    )
        
                scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
                
                model_params = {
                    **params_hyperparameter_model,
                    "Scheduler_Monitor": scheduler_monitor,
                    "objective": objective_hyperparameter_dataset}
                
                _RESULTS_.extend(hyperparameter_test(
                    model_params= model_params, 
                    dataset= dataset_hyperparameter_dataset,
                    repetitions = 5, n_parallel = N_PARALLEL,
                    to_epoch = to_epoch,
                    name = f"load_len:{load_len};val_len:{val_len}"))
                
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_load_len.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_load_len_raw.csv",sep=";")
       
        pars_overwrite_model[objective_str].update(
            hyperparameter_parse_name(_RESULTS_grouped.head(1).index[0]))
        pars_overwrite_model[objective_str]["load_len"] -= 48
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp.txt", 'w') as f:
             f.write(json.dumps(pars_overwrite_model[objective_str]))
        
        
    ############################################################################
    ############################################################################
    ##################  Hyperparameters: Preprocessing weekly  #################
    ############################################################################
    ############################################################################
    
    to_epoch = 24
    
    for objective_str in ["SDF","RP","VP", "VP_HARQ"]:
        # objective_str = "SDF"
        
        if objective_str in ["VP_HARQ","SDF","VP"]:
            continue
        
        ## load previous hyperparameter specs
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        
        ## create empty results list
        _RESULTS_ = []
        
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
            
        ## change objective specific settings
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": Loss_MSE_weighted(), "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":[
                        OBJ_P_weight_power(), OBJ_P_weights_stdmin()]})
            ascending= True
        
        
        ## parameter candidates
        candidates_learning_rate    = [np.exp(-3), np.exp(-5), np.exp(-7)]
        candidates_preprocessing    = [
            ["preprocessing:rank93",{"w":[PPB_rank_df(rolling=93,span=2), PPB_nan_to_0()]}],
            ["preprocessing:rank30",{"w":[PPB_rank_df(rolling=30,span=2), PPB_nan_to_0()]}],
            ["preprocessing:rank",{"w":[PPB_rank_df(rolling=0,span=2), PPB_nan_to_0()]}],
            ["preprocessing:norm93",{"w":[PPB_rank_df(rolling=93,span=2), PPB_nan_to_0()]}],
            ["preprocessing:norm30",{"w":[PPB_rank_df(rolling=30,span=2), PPB_nan_to_0()]}],
            ["preprocessing:norm",{"w":[PPB_rank_df(rolling=0,span=2), PPB_nan_to_0()]}],
            ["preprocessing:drop NA",{"w":[PPB_nan_drop_df()]}],
            ["preprocessing:None",{"w":[PPB_nan_to_0()]}]
            ]
        candidates_mode_validation       = ["rcv","vw"]
        
        for pp_units in candidates_preprocessing:
            for candidate_mode_validation in candidates_mode_validation:
                '''
                pp_units = ["preprocessing:rank93",{"w":[PPB_rank_df(rolling=93,span=2), PPB_nan_to_0()]}]
                candidate_mode_validation = "rcv"
                '''
                
                ## update standard parameters
                params_hyperparameter_dataset.update(pars_overwrite_model[objective_str])
                
                ## update from candidates
                params_hyperparameter_dataset.update({
                    "preprocessing":{'batchwise': pp_units[1], 'dataset': {'w': []},
                                     'filter': {'w': []}},
                    "mode_validation":candidate_mode_validation
                    })
                
                ## VP with HARQ columns
                if objective_str == "VP_HARQ":
                    use_columns = HARQ_columns
                    params_hyperparameter_structure["input_dim"] = (len(use_columns),1)
                elif pp_units[0] == "preprocessing:drop NA":
                    use_columns = DROP_NA_columns
                    params_hyperparameter_structure["input_dim"] = (len(use_columns),1)
                else:
                    use_columns = []
                    params_hyperparameter_structure["input_dim"] = (41,1)
                params_hyperparameter_model["structure"] = Layers_FFNN(params_hyperparameter_structure)
                
                ## load dataset
                params_hyperparameter_dataset["load_len"]+=48
                dataset_hyperparameter_dataset = PhD4_dataset(
                    **params_hyperparameter_dataset,
                    target_variable = target_variable,
                    use_columns= use_columns
                    )
        
                scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
                
                
                for learning_rate in candidates_learning_rate:
                    dataset_hyperparameter_dataset.shuffle()
                    
                    print_hint2(f"log(lr): {np.log(learning_rate)}, {pp_units[0]},"+\
                                " validation: {candidate_mode_validation}")
                    
                    model_params = {
                        **params_hyperparameter_model,
                        "Scheduler_Monitor": scheduler_monitor,
                        "learning_rate":learning_rate,
                        "objective": objective_hyperparameter_dataset}
                
                    _RESULTS_.extend(hyperparameter_test(
                        model_params= model_params, to_epoch = to_epoch,
                        dataset= dataset_hyperparameter_dataset,
                        repetitions = 5, n_parallel = N_PARALLEL,
                        name = f"{pp_units[0]};val_mode:{candidate_mode_validation};"+\
                            f"lr:{learning_rate}"))
        ## save the results and the hyperparameters
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_preprocessing.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_preprocessing_raw.csv",sep=";")
       
        pars_overwrite_model[objective_str].update(
            hyperparameter_parse_name(_RESULTS_grouped.head(1).index[0]))
        
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'w') as f:
             f.write(json.dumps(pars_overwrite_model[objective_str]))
        
    ## optimise hyperparameters
    
    ############################################################################
    ############################################################################
    ##############  Hyperparameters: Loss Function and Penalties  ##############
    ############################################################################
    ############################################################################
    
    ### The following settings have to be adjusted according to the actual results in the previous section
    ### maybe this section should be automised in the long term, automatically parsing the settings saved
    ### in the txt files in the hyperparameters folder
    
    to_epoch = 24
    
    preprocessing_units = {
        "SDF": {"w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()]},
        "RP": {"w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()]},
        "VP": {"w":[PPB_nan_to_0()]},
        "VP_HARQ": {"w":[PPB_nan_to_0()]}
        }
    
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-3)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    for objective_str in ["SDF","RP","VP", "VP_HARQ"]:
        # objective_str = "VP_HARQ"
        
        
        # if objective_str in ["RP"]:
        #     continue
        
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        pars_local_model = pars_overwrite_model[objective_str].copy()
        pars_local_model["preprocessing"] = \
            {'batchwise': preprocessing_units[objective_str], 'dataset': {'w': []}, 'filter': {'w': []}}
        pars_local_model["mode_validation"] = pars_local_model["val_mode"]
            
        ## empty results list
        _RESULTS_ = []
        
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        ## determine candidates for the optimisation problem
        if objective_str in ["RP","VP","VP_HARQ"]:
            ## losses
            candidate_loss_parameters = [
                ["base_penalty:Loss_MSE",Loss_MSE_weighted()],
                ["base_penalty:Loss_Huber0.1",Loss_Huber_weighted(threshold = 0.1)],
                ["base_penalty:Loss_Huber0.01",Loss_Huber_weighted(threshold = 0.01)],
                ["base_penalty:Loss_Huber0.001",Loss_Huber_weighted(threshold = 0.001)]
                ]
        else:
            candidate_loss_parameters = [
                ["penalty:weight_power+std_min",[OBJ_P_weight_power(), OBJ_P_weights_stdmin()]],
                ["penalty:weight_power",[OBJ_P_weight_power()]],
                ["penalty:std_min",[OBJ_P_weights_stdmin()]],
                ["penalty:long_short_ratio",[OBJ_P_long_short_ratio()]],
                ["penalty:non_zero",[OBJ_P_long_short_ratio()]]
                ]
            
        ## dataset handling
        params_hyperparameter_dataset.update(pars_local_model)
        
        ## VP with HARQ columns
        if objective_str == "VP_HARQ":
            use_columns = HARQ_columns
            params_hyperparameter_structure["input_dim"] = (len(use_columns),1)
        else:
            use_columns = []
            params_hyperparameter_structure["input_dim"] = (41,1)
        params_hyperparameter_model["structure"] = Layers_FFNN(params_hyperparameter_structure)
        
        ## load dataset
        params_hyperparameter_dataset["load_len"]+=48
        dataset_hyperparameter_dataset = PhD4_dataset(
            **params_hyperparameter_dataset,
            target_variable = target_variable,
            use_columns= use_columns
            )

        scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
        
        for candidate_loss_param in candidate_loss_parameters:
            
            print_hint2(candidate_loss_param[0])
            
            ## change objective specific settings
            if objective_str in ["RP","VP","VP_HARQ"]:
                objective_hyperparameter_dataset = RP_objective(
                    loss_fn = Loss_ENET,
                    loss_fn_par = {
                        "base_penalty": candidate_loss_param[1], "alpha_p"   : 0.00001, 
                        "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                    quantiles = 10, target_is_return=False if "VP" in objective_str else True)
                ascending= False
            else:
                objective_hyperparameter_dataset = SDF_objective(
                    loss_fn = Loss_SDF, loss_fn_par = {
                        "normalise":"1a", "return_mode":"simple",
                        "mode":"SDF", "penalties":candidate_loss_param[1]})
                ascending= True
            
            dataset_hyperparameter_dataset.shuffle()
            
            ## update parameters for model
            params_hyperparameter_model.update(model_parameter_updates[objective_str])
            
            model_params = {
                **params_hyperparameter_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_hyperparameter_dataset}
        
            _RESULTS_.extend(hyperparameter_test(
                model_params= model_params, to_epoch = to_epoch,
                dataset= dataset_hyperparameter_dataset,
                repetitions = 5, n_parallel = N_PARALLEL,
                name = f"{candidate_loss_param[0]}"))
                
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_loss_params.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_loss_params_raw.csv",sep=";")
       
        pars_overwrite_model[objective_str].update(
            hyperparameter_parse_name(_RESULTS_grouped.head(1).index[0]))
        
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp3.txt", 'w') as f:
             f.write(json.dumps(pars_overwrite_model[objective_str]))
                
    ############################################################################
    ############################################################################
    ############  Hyperparameters: Preprocessing other frequencies  ############
    ############################################################################
    ############################################################################
    
    to_epoch = 10
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {"w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()]},
        "RP": {"w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()]},
        "VP": {"w":[PPB_nan_to_0()]},
        "VP_HARQ": {"w":[PPB_nan_to_0()]}
        }
    
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-3)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    preprocessing_unit_defaults = {
        "m":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()],
        "q":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()]}
    
    ## define simple mixed frequency model with feed forward layers.
         
    hyperparameter_layers_multifreq = {
        "kernel_order": ["w","m","q"],
        "kernels":
            {"w":{"input_dim":(41,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                  "FFNN":[{"out_features":32}, {"out_features":16}],
                  "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                  "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                               {"activation":nn.ReLU(), "dropout":0.02},],
                  "TNN":[], "reshape":[True,True], "postprocessor":None, 
                  },
             "m":{"input_dim":(26,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                   "FFNN":[{"out_features":32}, {"out_features":16}],
                   "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                   "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                                {"activation":nn.ReLU(), "dropout":0.02},],
                   "TNN":[], "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay},
                   },
             "q":{"input_dim":(76,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                   "FFNN":[{"out_features":32}, {"out_features":16}],
                   "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                   "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                                {"activation":nn.ReLU(), "dropout":0.02},],
                   "TNN":[], "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay},
                   },
             },
    "FFNN":[
        {"out_features":24}, {"out_features":1},
        # {"out_features":20}, {"out_features":10}, {"out_features":1},
            ],
    "FFNN_sec":[{"activation":nn.ReLU(), "dropout":0.02},
                {"activation":nn.ReLU(), "dropout":0.02}]}
    
    params_hyperparameter_dataset.update({
        "input_frequencies": ["w","m","q"],
        "lookback":{"w":1,"m":1,"q":1}})
    ## test different preprocessing layers for both frequencies
    
    candidates_preprocessing_m = [
        ["preprocessing:rank183",{"m":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()]}],
        ["preprocessing:rank92",{"m":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()]}],
        ["preprocessing:rank",{"m":[PPB_rank_df(rolling=0,span=2), PPB_nan_to_0()]}],
        ["preprocessing:norm183",{"m":[PPB_norm_df(rolling=183,span=2), PPB_nan_to_0()]}],
        ["preprocessing:norm93",{"m":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}],
        ["preprocessing:norm",{"m":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()]}],
        ["preprocessing:None",{"m":[PPB_nan_to_0()]}]
        ]


    candidates_preprocessing_q = [
        ["preprocessing:rank366",{"q":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()]}],
        ["preprocessing:rank92",{"q":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()]}],
        ["preprocessing:rank",{"q":[PPB_rank_df(rolling=0,span=2), PPB_nan_to_0()]}],
        ["preprocessing:norm366",{"q":[PPB_norm_df(rolling=183,span=2), PPB_nan_to_0()]}],
        ["preprocessing:norm92",{"q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}],
        ["preprocessing:norm",{"q":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()]}],
        ["preprocessing:None",{"q":[PPB_nan_to_0()]}]
        ]

    for objective_str in ["RP","SDF","VP"]:
        ## objective_str = "VP"
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            ascending= True
            
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        ## empty results list
        _RESULTS_ = []
        use_columns = []
        
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        pars_local_model = pars_overwrite_model[objective_str].copy()
        
        
        for candidate in candidates_preprocessing_m:
            # candidate = candidates_preprocessing_m[0]
            
            preprocessing_local = {'batchwise': preprocessing_unit_defaults|{}, 
                                   'dataset': {'w': []}, 'filter': {'w': []}}
            preprocessing_local['batchwise'].update(preprocessing_units[objective_str])
            preprocessing_local['batchwise'].update(candidate[1])
            
            pars_local_model["preprocessing"] = preprocessing_local
            pars_local_model["mode_validation"] = pars_local_model["val_mode"]
            params_hyperparameter_dataset.update(pars_local_model)
            params_hyperparameter_dataset["load_len"]+=48
                
            params_hyperparameter_model["structure"] = Layers_multifreq(hyperparameter_layers_multifreq)
        
            ## load dataset
            dataset_hyperparameter_dataset = PhD4_dataset(
                **params_hyperparameter_dataset,
                target_variable = target_variable,
                use_columns = use_columns
                )
            dataset_hyperparameter_dataset.shuffle()
    
            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
            
            ## update parameters for model
            params_hyperparameter_model.update(model_parameter_updates[objective_str])
            params_hyperparameter_model.update({"model_type":ANN_MultiFreq})
                
            model_params = {
                **params_hyperparameter_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_hyperparameter_dataset}
        
            # model_local = model(**model_params)
            
            # model_local.train_validate(
            #     data_set=dataset_hyperparameter_dataset, to_epoch=to_epoch,
            #     fraction=1, frequency = "w",
            #     tolerance_relative = True, thread = 0)
        
            _RESULTS_.extend(hyperparameter_test(
                model_params= model_params, 
                dataset= dataset_hyperparameter_dataset,
                repetitions = 5, n_parallel = N_PARALLEL,
                name = f"m_{candidate[0]}", to_epoch = to_epoch))
        
        for candidate in candidates_preprocessing_q:
            
            preprocessing_local = {'batchwise': preprocessing_unit_defaults|{}, 
                                   'dataset': {'w': []}, 'filter': {'w': []}}
            preprocessing_local['batchwise'].update(preprocessing_units[objective_str])
            preprocessing_local['batchwise'].update(candidate[1])
            
            pars_local_model["preprocessing"] = preprocessing_local
            pars_local_model["mode_validation"] = pars_local_model["val_mode"]
            params_hyperparameter_dataset.update(pars_local_model)
            params_hyperparameter_dataset["load_len"]+=48
                
            params_hyperparameter_model["structure"] = Layers_multifreq(hyperparameter_layers_multifreq)
        
            ## load dataset
            dataset_hyperparameter_dataset = PhD4_dataset(
                **params_hyperparameter_dataset,
                target_variable = target_variable,
                use_columns = use_columns
                )
            dataset_hyperparameter_dataset.shuffle()
    
            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
            
            ## update parameters for model
            params_hyperparameter_model.update(model_parameter_updates[objective_str])
            params_hyperparameter_model.update({"model_type":ANN_MultiFreq})
                
            model_params = {
                **params_hyperparameter_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_hyperparameter_dataset}
            _RESULTS_.extend(hyperparameter_test(
                model_params= model_params, 
                dataset= dataset_hyperparameter_dataset,
                repetitions = 5, n_parallel = N_PARALLEL,
                name = f"q_{candidate[0]}", to_epoch = to_epoch))
        
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"mq_{objective_str}_{variant}_mixfreq_pp.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_mixfreq_pp_raw.csv",sep=";")
       
        pars_overwrite_model[objective_str].update(
            hyperparameter_parse_name(_RESULTS_grouped.head(1).index[0]))
        
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp4.txt", 'w') as f:
              f.write(json.dumps(pars_overwrite_model[objective_str]))

if False:
    ###########################################################################
    ###########################################################################
    ##########################  Activation Functions  #########################
    ###########################################################################
    ###########################################################################
    
    to_epoch = 10
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {"w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()]},
        "RP": {"w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()]},
        "VP": {"w":[PPB_nan_to_0()]},
        "VP_HARQ": {"w":[PPB_nan_to_0()]}
        }
    
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-3)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    '''
    Grid search for all feed forward models
    
    '''
    candidates_activations = [
        ['DoubleTanh',act_DoubleTanh()],
        ["Hardtanh",nn.Hardtanh()],
        ["LeakyReLU",nn.LeakyReLU(0.1)],
        ["Softsign",nn.Softsign()],
        ["Sigmoid",nn.Sigmoid()],
        ["Softplus",nn.Softplus()],
        ["Tanh",nn.Tanh()],
        ["Mish",nn.Mish()],
        ["ELU",nn.ELU()]]

    for objective_str in ["SDF","RP","VP", "VP_HARQ"]:
        # objective_str = "RP"
        
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            ascending= True
        
            
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        
        ## empty results list
        _RESULTS_ = []
        use_columns = []
        
        
        ## update dataset settings from hyperparameter file
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        # pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model["val_mode"]
        params_hyperparameter_dataset.update(pars_overwrite_model[objective_str])
        params_hyperparameter_dataset["mode_validation"] = params_hyperparameter_dataset["val_mode"]
        
        
        # further objective specific settings: input columns and input dim
        if objective_str == "VP_HARQ":
            use_columns = HARQ_columns
            params_hyperparameter_structure["input_dim"] = (len(use_columns),1)
        else:
            use_columns = []
            params_hyperparameter_structure["input_dim"] = (41,1)
        
        
        ## dataset import
        params_hyperparameter_dataset.update(
            {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                              'dataset': {'w': []}, 'filter': {'w': []}}})
        params_hyperparameter_dataset["load_len"]+=48
        dataset_hyperparameter_dataset = PhD4_dataset(
            **params_hyperparameter_dataset,
            target_variable = target_variable,
            use_columns = use_columns)
        dataset_hyperparameter_dataset.shuffle()
        
        for activation in candidates_activations:
        
            
            ## create structure
            local_structure = copy.deepcopy(params_hyperparameter_structure)
            local_structure["FFNN_sec"] = [
                {"activation":activation[1],"dropout":.02}]*3
            
            params_hyperparameter_model["structure"] = Layers_FFNN(local_structure)
            
            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
            
            ## update parameters for model
            params_hyperparameter_model.update(model_parameter_updates[objective_str])
            params_hyperparameter_model.update({"model_type":FFNN})
            
            model_params = {
                **params_hyperparameter_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_hyperparameter_dataset}
            _RESULTS_.extend(hyperparameter_test(
                model_params= model_params, 
                dataset= dataset_hyperparameter_dataset,
                repetitions = 5, n_parallel = N_PARALLEL,
                name = f"{activation[0]}",
                to_epoch = to_epoch))
            ## test structure
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_activations.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_activations_raw.csv",sep=";")
    
    ############################################################################
    ############################################################################
    #####################  Hyperparameters: Linear Models  #####################
    ############################################################################
    ############################################################################    
    
    to_epoch = 10
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {"w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()]},
        "RP": {"w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()]},
        "VP": {"w":[PPB_nan_to_0()]},
        "VP_HARQ": {"w":[PPB_nan_to_0()]}
        }
    
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-3)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    '''
    Grid search for all feed forward models
    
    '''
    
    candidates_n_layers = [2,3,4]
    candidates_n_neurons = [128,64,32]
    candidates_activations = {
        "SDF":[['DoubleTanh',act_DoubleTanh()], ["ELU",nn.ELU()], 
               ["LeakyReLU",nn.LeakyReLU(0.1)], ["Softplus",nn.Softplus()]],
        "RP":[['LeakyReLU',nn.LeakyReLU(0.1)], ["Mish",nn.Mish()], 
              ["Hardtanh",nn.Hardtanh()], ["DoubleTanh",act_DoubleTanh()]],
        "VP":[['ELU',nn.ELU()], ["Softsign",nn.Softsign()], 
              ["Tanh",nn.Tanh()], ["Hardtanh",nn.Hardtanh()]],
        "VP_HARQ":[['ELU',nn.ELU()], ["Softsign",nn.Softsign()], 
                   ["Tanh",nn.Tanh()], ["Hardtanh",nn.Hardtanh()]]}
    cancidates_dropout = [0,.05,.1]

    for objective_str in ["SDF","RP","VP", "VP_HARQ"]:
        # objective_str = "RP"
        
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            ascending= True
        
            
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        
        ## empty results list
        _RESULTS_ = []
        use_columns = []
        
        
        ## update dataset settings from hyperparameter file
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        # pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model["val_mode"]
        params_hyperparameter_dataset.update(pars_overwrite_model[objective_str])
        params_hyperparameter_dataset["mode_validation"] = params_hyperparameter_dataset["val_mode"]
        
        
        # further objective specific settings: input columns and input dim
        if objective_str == "VP_HARQ":
            use_columns = HARQ_columns
            params_hyperparameter_structure["input_dim"] = (len(use_columns),1)
        else:
            use_columns = []
            params_hyperparameter_structure["input_dim"] = (41,1)
        
        
        ## dataset import
        params_hyperparameter_dataset.update(
            {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                              'dataset': {'w': []}, 'filter': {'w': []}}})
        params_hyperparameter_dataset["load_len"]+=48
        dataset_hyperparameter_dataset = PhD4_dataset(
            **params_hyperparameter_dataset,
            target_variable = target_variable,
            use_columns = use_columns)
        dataset_hyperparameter_dataset.shuffle()
        
        for n_layers in candidates_n_layers:
            for n_neurons in candidates_n_neurons:
                for activation in candidates_activations[objective_str]:
                    for dropout in cancidates_dropout:
                    
                        print_hint2(f"{n_layers}-{n_neurons}-{activation[0]}-{dropout}",
                                    width_field = 50)
                        
                        ## create structure
                        local_structure = params_hyperparameter_structure.copy()
                        local_structure["FFNN"] = []
                        local_structure["FFNN_sec"] = []
                        for layer_n in range(n_layers):
                            local_structure["FFNN"].append({
                                "out_features":int(n_neurons/2**n_layers)})
                            local_structure["FFNN_sec"].append({
                                "activation":activation[1],"dropout":dropout})
                        local_structure["FFNN"].append({"out_features":1})
                        
                        params_hyperparameter_model["structure"] = Layers_FFNN(local_structure)
                        
                        scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
                        
                        ## update parameters for model
                        params_hyperparameter_model.update(model_parameter_updates[objective_str])
                        params_hyperparameter_model.update({"model_type":FFNN})
                        
                        model_params = {
                            **params_hyperparameter_model,
                            "Scheduler_Monitor": scheduler_monitor,
                            "objective": objective_hyperparameter_dataset}
                        _RESULTS_.extend(hyperparameter_test(
                            model_params= model_params, 
                            dataset= dataset_hyperparameter_dataset,
                            repetitions = 5, n_parallel = N_PARALLEL,
                            name = f"{n_layers}_{n_neurons}_{activation[0]}_{dropout}",
                            to_epoch = to_epoch))
                        ## test structure
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_w.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_w_raw.csv",sep=";")

if False:
    ############################################################################
    ############################################################################
    #####################  Hyperparameters: FFNN wmq mixed  ####################
    ############################################################################
    ############################################################################
 
    to_epoch = 10
    
    ## parameters from previous stages
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-3)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {
            "w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]},
        "RP": {
            "w":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()],
            "m":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=366,span=2), PPB_nan_to_0()]},
        "VP": {
            "w":[PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}}
    
    structures_w = {
        "SDF":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "RP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "VP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2}}
    
    structures_m = {
        "SDF":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "RP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "VP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2}}
    
    structures_q = {
        "SDF":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "RP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "VP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2}}
    
    structures_joiner = {
        "SDF":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "RP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "VP":{
            "FFNN":[{"out_features":32}, {"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2}}
    
    structure_default =  {
        "kernel_order": ["w","m","q"],
        "kernels": {
            "w":{"input_dim":(41,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                 "FFNN":[{"out_features":32}, {"out_features":16}],
                 "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                 "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                              {"activation":nn.ReLU(), "dropout":0.02},],
                 "TNN":[], "reshape":[True,True], "postprocessor":None},
             "m":{"input_dim":(26,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                  "FFNN":[{"out_features":32}, {"out_features":16}],
                  "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                  "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                               {"activation":nn.ReLU(), "dropout":0.02},],
                  "TNN":[], "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay}},
             "q":{"input_dim":(76,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                  "FFNN":[{"out_features":32}, {"out_features":16}],
                  "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                  "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                               {"activation":nn.ReLU(), "dropout":0.02},],
                  "TNN":[], "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay}}},
    "FFNN":[
        {"out_features":16}, {"out_features":1},
        # {"out_features":20}, {"out_features":10}, {"out_features":1},
            ],
    "FFNN_sec":[{"activation":nn.ReLU(), "dropout":0.02},
                {"activation":nn.ReLU(), "dropout":0.02}]}
    
    ## parameters for the dataset need to be upgraded at this point
    params_hyperparameter_dataset.update({
        "input_frequencies": ["w","m","q"],
        "lookback":{"w":1,"m":1,"q":1}})
    
    candidates_n_layers = [1,2,3,4]
    candidates_n_neurons = [64,32,16]
    candidates_activations = {
        "SDF":[['DoubleTanh',act_DoubleTanh()], ["ELU",nn.ELU()], ["LeakyReLU",nn.LeakyReLU()]],
        "RP":[['LeakyReLU',nn.LeakyReLU()], ["Mish",nn.Mish()], ["Hardtanh",nn.Hardtanh()]],
        "VP":[['ELU',nn.ELU()], ["Softsign",nn.Softsign()], ["Tanh",nn.Tanh()]],
        "VP_HARQ":[['ELU',nn.ELU()], ["Softsign",nn.Softsign()], 
                   ["Tanh",nn.Tanh()], ["Hardtanh",nn.Hardtanh()]]}
    cancidates_dropout = [.01,.05,.2]
    
    test_str = ""
    
    ## test_setup
    # candidates_n_layers = [1];  candidates_n_neurons = [16]
    # candidates_activations = {
    #     "SDF":[["LeakyReLU",nn.LeakyReLU()]], "RP":[["Hardtanh",nn.Hardtanh()]],
    #     "VP":[["Tanh",nn.Tanh()]], "VP_HARQ":[["Hardtanh",nn.Hardtanh()]]}
    # cancidates_dropout = [.01]
    # test_str = "test"
    
    
    ## optimise monthly and quarterly frequency model part
    for objective_str in ["SDF","RP","VP"]:
        
        if objective_str not in ["VP"]:
            continue ### !!! skip clause
        # objective_str = "VP"
        
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            ascending= True
        
            
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        
        ## empty results list
        _RESULTS_ = []
        use_columns = []
        
        
        ## update dataset settings from hyperparameter file
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        # pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model["val_mode"]
        params_hyperparameter_dataset.update(pars_overwrite_model[objective_str])
        params_hyperparameter_dataset["mode_validation"] = params_hyperparameter_dataset["val_mode"]
        
        
        ## dataset import
        params_hyperparameter_dataset.update(
            {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                              'dataset': {'w': [],"m":[],"q":[]}, 
                              'filter': {'w': [],"m":[],"q":[]}}})
        params_hyperparameter_dataset["meta_series"] = {"m":"m_dist","q":"q_dist"}
        params_hyperparameter_dataset["load_len"]+=48
        dataset_hyperparameter_dataset = PhD4_dataset(
            **params_hyperparameter_dataset,
            target_variable = target_variable,
            use_columns = use_columns)
        dataset_hyperparameter_dataset.shuffle()
        
        
        ## optimise frequency joiner    
        # _RESULTS_ = _RESULTS_.values.tolist()
        _RESULTS_ = []
        for n_neurons in candidates_n_neurons:
            for n_layers in candidates_n_layers:
                for activation in candidates_activations[objective_str]:
                    for dropout in cancidates_dropout:
                            # n_layers = 3; activation= ["Softsign",nn.Softsign()]; dropout = .2;
                            
                            local_structure = structure_default.copy()
                            local_structure["kernels"]["w"].update(structures_w[objective_str])
                            local_structure["kernels"]["m"].update(structures_m[objective_str])
                            local_structure["kernels"]["q"].update(structures_q[objective_str])
                            
                            local_structure["FFNN"] = []
                            local_structure["FFNN_sec"] = []
                            for layer_n in range(n_layers):
                                local_structure["FFNN"].append({
                                    "out_features":int(n_neurons/2**layer_n)})
                                local_structure["FFNN_sec"].append({
                                    "activation":activation[1],"dropout":dropout})
                            local_structure["FFNN"].append({"out_features":1})
                            
                            params_hyperparameter_model["structure"] = Layers_multifreq(local_structure)
                            
                            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
                            
                            ## update parameters for model
                            params_hyperparameter_model.update(model_parameter_updates[objective_str])
                            params_hyperparameter_model.update({"model_type":ANN_MultiFreq})
                            
                            model_params = {
                                **params_hyperparameter_model,
                                "Scheduler_Monitor": scheduler_monitor,
                                "objective": objective_hyperparameter_dataset}
                            _RESULTS_.extend(hyperparameter_test(
                                model_params= model_params, 
                                dataset= dataset_hyperparameter_dataset,
                                repetitions = 5, n_parallel = N_PARALLEL,
                                name = f"joiner_{n_layers}_{n_neurons}_{activation[0]}_{dropout}",
                                to_epoch = to_epoch))
                            ## test structure
        
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        frequency_, n_layers, n_neurons, activation, dropout = _RESULTS_grouped.index.values[0].split("_")
        del frequency_
        n_layers = int(n_layers); n_neurons = int(n_neurons); dropout = float(dropout);
        activation = [i[1] for i in candidates_activations[objective_str] if i[0] == activation][0]
        structures_joiner[objective_str]["FFNN"] = [
            {"out_features":int(n_neurons*2**(-layer))} for layer in range(n_layers)]
        structures_joiner[objective_str]["FFNN"].append({"out_features":1})
        structures_joiner[objective_str]["FFNN_sec"]= [
            {"activation":activation,"dropout":dropout}]*(n_layers)
        
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_wmq_multi_joiner{test_str}.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_wmq_multi_raw_joiner{test_str}.csv",sep=";")
        
        for frequency in ["w","m","q"]:
            # if frequency == "m":
            #     continue
            # frequency = "m"; 
            sub_results = []
            for n_layers in candidates_n_layers:
                for n_neurons in candidates_n_neurons:
                    for activation in candidates_activations[objective_str]:
                        for dropout in cancidates_dropout:
                            # n_layers = 3; n_neurons = 64; activation= ["Softsign",nn.Softsign()]; dropout = .05;
                            
                            ## create structure
                            local_structure = structure_default.copy()
                            local_structure["kernels"]["w"].update(structures_w[objective_str])
                            local_structure["kernels"]["m"].update(structures_m[objective_str])
                            local_structure["kernels"]["q"].update(structures_q[objective_str])
                            local_structure.update(structures_joiner)
                            
                            local_structure["kernels"][frequency]["FFNN"] = []
                            local_structure["kernels"][frequency]["FFNN_sec"] = []
                            for layer_n in range(n_layers):
                                local_structure["kernels"][frequency]["FFNN"].append({
                                    "out_features":int(n_neurons/2**n_layers)})
                                local_structure["kernels"][frequency]["FFNN_sec"].append({
                                    "activation":activation[1],"dropout":dropout})
                            
                            params_hyperparameter_model["structure"] = Layers_multifreq(local_structure)
                            
                            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
                            
                            ## update parameters for model
                            params_hyperparameter_model.update(model_parameter_updates[objective_str])
                            params_hyperparameter_model.update({"model_type":ANN_MultiFreq})
                            
                            model_params = {
                                **params_hyperparameter_model,
                                "Scheduler_Monitor": scheduler_monitor,
                                "objective": objective_hyperparameter_dataset}
                            sub_results.extend(hyperparameter_test(
                                model_params= model_params, 
                                dataset= dataset_hyperparameter_dataset,
                                repetitions = 5, n_parallel = N_PARALLEL,
                                name = f"{frequency}_{n_layers}_{n_neurons}_{activation[0]}_{dropout}",
                                to_epoch = to_epoch))
            ## update structures_[frequency]
            # _RESULTS_.extend(sub_results.copy())
            sub_results = pd.DataFrame(sub_results).groupby("name").mean().sort_values("test_score",ascending=ascending)
            frequency_, n_layers, n_neurons, activation, dropout = sub_results.index.values[0].split("_")
            del frequency_
            n_layers = int(n_layers); n_neurons = int(n_neurons); dropout = float(dropout);
            activation = [i[1] for i in candidates_activations[objective_str] if i[0] == activation][0]
            if frequency =="w":
                structures_w[objective_str]["FFNN"] = [
                    {"out_features":int(n_neurons*2**(-layer))} for layer in range(n_layers)]
                structures_w[objective_str]["FFNN_sec"]= [
                    {"activation":activation,"dropout":dropout}]*(n_layers)
            elif frequency =="m":
                structures_m[objective_str]["FFNN"] = [
                    {"out_features":int(n_neurons*2**(-layer))} for layer in range(n_layers)]
                structures_m[objective_str]["FFNN_sec"]= [
                    {"activation":activation,"dropout":dropout}]*(n_layers)
            else:
                structures_q[objective_str]["FFNN"] = [
                    {"out_features":int(n_neurons*2**(-layer))} for layer in range(n_layers)]
                structures_q[objective_str]["FFNN_sec"]= [
                    {"activation":activation,"dropout":dropout}]*(n_layers)
            
            sub_results.to_csv(
                HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_wmq_multi_{frequency}{test_str}.csv",sep=";")
            sub_results.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_wmq_multi_raw_{frequency}{test_str}.csv",sep=";")
    
    
    
    
if False:    
    ############################################################################
    ############################################################################
    ####################  Hyperparameters: FFNN wmq joined  ####################
    ############################################################################
    ############################################################################
    
    to_epoch = 10
    
    ## parameters from previous stages
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-3)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {
            "w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]},
        "RP": {
            "w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0(), PPB_nan_to_0()],
            "m":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=366,span=2), PPB_nan_to_0()]},
        "VP": {
            "w":[PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}}
    
    params_joined_FFNN_structure = params_hyperparameter_structure |{}
    params_joined_FFNN_structure["input_dim"] = 143
    
    params_hyperparameter_dataset.update({
        "input_frequencies": ["w","m","q"],
        "lookback":{"w":1,"m":1,"q":1}})
    
    candidates_n_layers = [2,3,4]
    candidates_n_neurons = [128,64,32]
    candidates_activations = {
        "SDF":[['DoubleTanh',act_DoubleTanh()], ["ELU",nn.ELU()], 
               ["LeakyReLU",nn.LeakyReLU(0.1)], ["Softplus",nn.Softplus()]],
        "RP":[['LeakyReLU',nn.LeakyReLU(0.1)], ["Mish",nn.Mish()], 
              ["Hardtanh",nn.Hardtanh()], ["DoubleTanh",act_DoubleTanh()]],
        "VP":[['ELU',nn.ELU()], ["Softsign",nn.Softsign()], 
              ["Tanh",nn.Tanh()], ["Hardtanh",nn.Hardtanh()]],
        "VP_HARQ":[['ELU',nn.ELU()], ["Softsign",nn.Softsign()], 
                   ["Tanh",nn.Tanh()], ["Hardtanh",nn.Hardtanh()]]}
    cancidates_dropout = [0,.05,.1]

    for objective_str in ["SDF","RP","VP"]:
        # objective_str = "RP" # objective_str = "SDF" # objective_str = "RP"
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            ascending= True
        
            
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        
        ## empty results list
        _RESULTS_ = []
        use_columns = []
        
        
        ## update dataset settings from hyperparameter file
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        # pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model["val_mode"]
        params_hyperparameter_dataset.update(pars_overwrite_model[objective_str])
        params_hyperparameter_dataset["mode_validation"] = params_hyperparameter_dataset["val_mode"]
        
        
        ## dataset import
        params_hyperparameter_dataset.update(
            {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                              'dataset': {'w': [],"m":[],"q":[]}, 
                              'filter': {'w': [],"m":[],"q":[]}}})
        params_hyperparameter_dataset["load_len"]+=48
        params_hyperparameter_dataset["merge_freq"]=True
        dataset_hyperparameter_dataset = PhD4_dataset(
            **params_hyperparameter_dataset,
            target_variable = target_variable,
            use_columns = use_columns)
        dataset_hyperparameter_dataset.shuffle()
        
        for n_layers in candidates_n_layers:
            for n_neurons in candidates_n_neurons:
                for activation in candidates_activations[objective_str]:
                    for dropout in cancidates_dropout:
                    
                        
                        ## create structure
                        local_structure = params_joined_FFNN_structure.copy()
                        local_structure["FFNN"] = []
                        local_structure["FFNN_sec"] = []
                        for layer_n in range(n_layers):
                            local_structure["FFNN"].append({
                                "out_features":int(n_neurons/2**(n_layers+1))})
                            local_structure["FFNN_sec"].append({
                                "activation":activation[1],"dropout":dropout})
                        local_structure["FFNN"].append({"out_features":1})
                        
                        params_hyperparameter_model["structure"] = Layers_FFNN(local_structure)
                        
                        scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
                        
                        ## update parameters for model
                        params_hyperparameter_model.update(model_parameter_updates[objective_str])
                        params_hyperparameter_model.update({"model_type":FFNN})
                        
                        model_params = {
                            **params_hyperparameter_model,
                            "Scheduler_Monitor": scheduler_monitor,
                            "objective": objective_hyperparameter_dataset}
                        _RESULTS_.extend(hyperparameter_test(
                            model_params= model_params, 
                            dataset= dataset_hyperparameter_dataset,
                            repetitions = 5, n_parallel = N_PARALLEL,
                            name = f"{n_layers}_{n_neurons}_{activation[0]}_{dropout}",
                            to_epoch = to_epoch))
                        ## test structure
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_wmw_joined.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_FFNN_wmw_joined_raw.csv",sep=";")
    
    
    
    ############################################################################
    ############################################################################
    #########################  Lookback Period: Weekly  ########################
    ############################################################################
    ############################################################################
    
    to_epoch = 10
    
    ## parameters from previous stages
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-5)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {
            "w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]},
        "RP": {
            "w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()],
            "m":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=366,span=2), PPB_nan_to_0()]},
        "VP": {
            "w":[PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}}
    
    structure_lookback_w = {
        "input_dim":(41,6), "in_channels":1, "lookback_dim":6,
        "reshape": [True,True],
        "CNN":[
            {"in_channels":1, "out_channels":3, "kernel_size":(1,4),
              "stride":(1,1), "padding":(0,0), "dilation":1},
            {"in_channels":1, "out_channels":24, "kernel_size":(41,1), 
             "stride":(1,1), "padding":(0,0), "dilation":1}],
        "CNN_sec":[
            {"activation":nn.Softsign(),"dropout":.05}, 
            {"activation":nn.Softsign(),"dropout":.05}],
        "FFNN":[{"out_features":40}, {"out_features":20}, {"out_features":1}],
        "CNN_norm":nn.BatchNorm2d, "FFNN_norm":nn.BatchNorm1d, "wavelength":0,
        "FFNN_sec":[{"activation":nn.Softsign(), "dropout":.02},
                    {"activation":nn.Softsign(), "dropout":.02}],
        }
    structure_lookback_w["TNN"] = [ 
        {"d_model":24, "nhead":6, "dim_feedforward": 64, "dropout":.1, "activation":nn.Softsign()},
        ]
    
    ## w lookback experiment
    # only w data
    w_lookback_candidates = [4,9,13]
    
    ## first just weekly data.
    
    for objective_str in ["SDF","RP","VP"]:
        # objective_str = "RP"
        
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            ascending= True
        
            
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        
        ## empty results list
        _RESULTS_ = []
        use_columns = []
        
        
        ## update dataset settings from hyperparameter file
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        # pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model["val_mode"]
        params_hyperparameter_dataset.update(pars_overwrite_model[objective_str])
        params_hyperparameter_dataset["mode_validation"] = params_hyperparameter_dataset["val_mode"]
        params_hyperparameter_dataset["load_len"]+=48
        params_hyperparameter_dataset["merge_freq"]=False
        
        for candidate_lookback in w_lookback_candidates:
            # candidate_lookback = 4
            
            ## dataset import
            params_hyperparameter_dataset.update(
                {"preprocessing":{'batchwise': {"w":preprocessing_units[objective_str]["w"]}, 
                                  'dataset': {'w': []}, 
                                  'filter': {'w': []}},
                 "lookback":{"w":[candidate_lookback,1]}})
            
            dataset_hyperparameter_dataset = PhD4_dataset(
                **params_hyperparameter_dataset,
                target_variable = target_variable,
                use_columns = use_columns)
            
            dataset_hyperparameter_dataset.shuffle()
            
            structure_lookback_w["CNN"][0].update(
                {"kernel_size":(1,candidate_lookback),"out_channels":(candidate_lookback//2+1)})
            structure_lookback_w["input_dim"] = (
                structure_lookback_w["input_dim"][0],candidate_lookback)
            structure_lookback_w["lookback_dim"] = candidate_lookback
            params_hyperparameter_model["structure"] = Layers_TNN_CNN_FFNN(Layers_FFNN_CNN(structure_lookback_w).structure)
            
            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
            
            ## update parameters for model
            params_hyperparameter_model.update(model_parameter_updates[objective_str])
            params_hyperparameter_model.update({"model_type":CNN_TNN})
            
            model_params = {
                **params_hyperparameter_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_hyperparameter_dataset}
            _RESULTS_.extend(hyperparameter_test(
                model_params= model_params, 
                dataset= dataset_hyperparameter_dataset,
                repetitions = 5, n_parallel = N_PARALLEL,
                name = f"LB_w_{candidate_lookback}",
                to_epoch = to_epoch))
            ## test structure
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(
            HYPERPARAMETER_PATH+f"{objective_str}_{variant}_LB_w.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_LB_w_raw.csv",sep=";")
            
    ## then monthly and quarterly data

    ############################################################################
    ############################################################################
    ###################  Lookback Period: Monthly quarterly  ###################
    ############################################################################
    ############################################################################
    
    to_epoch = 10
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {
            "w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]},
        "RP": {
            "w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0(), PPB_nan_to_0()],
            "m":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=366,span=2), PPB_nan_to_0()]},
        "VP": {
            "w":[PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}}
    
    model_parameter_updates = {
        "SDF": {"learning_rate": 0.0009118819655545162},
        "RP": {"learning_rate": 0.006737946999085467},
        "VP": {"learning_rate": 0.006737946999085467},
        "VP_HARQ": {"learning_rate": 0.006737946999085467},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    structure_lookback_wmq = {
         "kernel_order": ["w","m","q"],
         "kernels":
             {"w":{"input_dim":(41,4),"in_channels":1, "CNN":[
                   {"in_channels":1, "out_channels":3, "kernel_size":(1,4), 
                    "stride":(1,1), "padding": (0,0),"dilation":1},
                   {"in_channels":1, "out_channels":24, "kernel_size":(41,1), 
                    "stride":(1,1), "padding": (0,0),"dilation":1}],
                   "CNN_sec": [{"activation":nn.Softsign()}, {"activation":nn.Softsign()}],
                   "FFNN":[{"out_features":50}, {"out_features":10}],
                   "CNN_norm":nn.BatchNorm2d, "FFNN_norm":BatchNorm_self,
                   "FFNN_sec": [{"activation":nn.Softsign(), "dropout":0.02},
                                {"activation":nn.Softsign(), "dropout":0.02},],
                   "TNN":[{"d_model":24, "nhead":6, "dim_feedforward": 32,
                    "dropout":0.2, "activation":nn.Softsign()}],
                   "reshape":[True,True], "postprocessor":None, 
                   },
              "m":{"input_dim":(26,3),"in_channels":1, "CNN":[
                    {"in_channels":1, "out_channels":3, "kernel_size":(1,3), 
                     "stride":(1,1), "padding": (0,0),"dilation":1},
                    {"in_channels":1, "out_channels":16, "kernel_size":(26,1), 
                     "stride":(1,1), "padding": (0,0),"dilation":1}],
                    "CNN_sec": [{"activation":nn.Softsign()}, {"activation":nn.Softsign()}],
                    "FFNN":[{"out_features":30}, {"out_features":10}],
                    "CNN_norm":nn.BatchNorm2d, "FFNN_norm":BatchNorm_self,
                    "FFNN_sec": [{"activation":nn.Softsign(), "dropout":0.02},
                                 {"activation":nn.Softsign(), "dropout":0.02},],
                    "TNN":[{"d_model":16, "nhead":4, "dim_feedforward": 32,
                     "dropout":0.2, "activation":nn.Softsign()}],
                    "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay},
                    },
              "q":{"input_dim":(76,4),"in_channels":1, "CNN":[
                    {"in_channels":1, "out_channels":3, "kernel_size":(1,4), 
                     "stride":(1,1), "padding": (0,0),"dilation":1},
                    {"in_channels":1, "out_channels":32, "kernel_size":(76,1), 
                     "stride":(1,1), "padding": (0,0),"dilation":1}],
                    "CNN_sec": [{"activation":nn.Softsign()}, {"activation":nn.Softsign()}],
                    "FFNN":[{"out_features":50}, {"out_features":10}],
                    "CNN_norm":nn.BatchNorm2d, "FFNN_norm":BatchNorm_self,
                    "FFNN_sec": [{"activation":nn.Softsign(), "dropout":0.02},
                                 {"activation":nn.Softsign(), "dropout":0.02},],
                    "TNN":[{"d_model":32, "nhead":8, "dim_feedforward": 32,
                     "dropout":0.2, "activation":nn.Softsign()}],
                    "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay}}},
     "FFNN":[
         {"out_features":15}, {"out_features":1}],
     "FFNN_sec":[{"activation":nn.Softsign(), "dropout":0.02},
                 {"activation":nn.Softsign(), "dropout":0.02}],
     }
    
        
    ## m lookback
    lookback_candidates = {"m":[3,6,12],"q":[4,8]}
    
    standard_lookback = {
        "w":[4,1],"m":[3,1],"q":[4,1]}
    
    for objective_str in ["SDF","RP","VP"]:
        # objective_str = "RP"
        
        if objective_str in ["RP","VP","VP_HARQ"]:
            objective_hyperparameter_dataset = RP_objective(
                loss_fn = Loss_ENET,
                loss_fn_par = {
                    "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                    "lambda_p"  : 0.000000001 , "volatility":True if "VP" in objective_str else False},
                quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            ascending= False
        else:
            objective_hyperparameter_dataset = SDF_objective(
                loss_fn = Loss_SDF, loss_fn_par = {
                    "normalise":"1a", "return_mode":"simple",
                    "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            ascending= True
        
            
        ## load target variable for frequency
        target_variable = TARGET_VARIABLES[objective_str]
        
        
        ## empty results list
        _RESULTS_ = []
        use_columns = []
        
        
        ## update dataset settings from hyperparameter file
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        # pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model["val_mode"]
        params_hyperparameter_dataset.update(pars_overwrite_model[objective_str])
        params_hyperparameter_dataset["mode_validation"] = params_hyperparameter_dataset["val_mode"]
        params_hyperparameter_dataset["load_len"]+=48
        params_hyperparameter_dataset["merge_freq"]=False
        for frequency in ["m","q"]:
            for candidate_lookback in lookback_candidates[frequency]:
                # frequency = "m"; candidate_lookback = lookback_candidates[frequency][0]
                
                ## dataset import
                lookback = standard_lookback.copy()
                lookback[frequency][0] = candidate_lookback 
                params_hyperparameter_dataset.update(
                    {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                                      'dataset': {'w': []}, 
                                      'filter': {'w': []}},
                     "lookback":lookback,
                     "input_frequencies":["w","m","q"]})
                
                dataset_hyperparameter_dataset = PhD4_dataset(
                    **params_hyperparameter_dataset,
                    target_variable = target_variable,
                    use_columns = use_columns)
                
                dataset_hyperparameter_dataset.shuffle()
                
                structure_lookback_copy = structure_lookback_wmq.copy()
                structure_lookback_copy["kernels"][frequency]["CNN"][0].update(
                    {"kernel_size":(1,candidate_lookback),"out_channels":(candidate_lookback//2+1)})
                structure_lookback_copy["kernels"][frequency]["input_dim"] = (
                    structure_lookback_copy["kernels"][frequency]["input_dim"][0],candidate_lookback)
                structure_lookback_copy["kernels"][frequency]["lookback_dim"] = candidate_lookback
                
                
                params_hyperparameter_model["structure"] = Layers_multifreq(structure_lookback_copy)
                
                scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
                
                ## update parameters for model
                params_hyperparameter_model.update(model_parameter_updates[objective_str])
                params_hyperparameter_model.update({"model_type":ANN_MultiFreq})
                
                model_params = {
                    **params_hyperparameter_model,
                    "Scheduler_Monitor": scheduler_monitor,
                    "objective": objective_hyperparameter_dataset}
                _RESULTS_.extend(hyperparameter_test(
                    model_params= model_params, 
                    dataset= dataset_hyperparameter_dataset,
                    repetitions = 5, n_parallel = N_PARALLEL,
                    name = f"LB_wmq_{frequency}_{candidate_lookback}",
                    to_epoch = to_epoch))
                ## test structure
        _RESULTS_ = pd.DataFrame(_RESULTS_)
        _RESULTS_grouped = _RESULTS_.groupby("name").mean().sort_values("test_score",ascending=ascending)
        _RESULTS_grouped.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_LB_mq.csv",sep=";")
        _RESULTS_.to_csv(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_LB_mq_raw.csv",sep=";")
    
    
####################################################################################################
####################################################################################################
####################################################################################################
########################   Neural Network Empirical Applications (Stage 3)   #######################
####################################################################################################
####################################################################################################
####################################################################################################
'''
Rolling window estimation for models included in the analysis

Models to be estimated:
    
    FF_w
    FF_wmq
    
    CT_LB_w
    CT_LB_wmq
    
    for 
    
    RP
    SDF
    VP
    VP-HARQ
     
'''

train_validate_pars = {
    "to_epoch":100,
    "fraction":1,
    "frequency": "w"}

fc_pars = {
    "date_entity":"date",
    "return_mode":"simple"
    }
test_pars = {
    "frequency":"w"
    }
if False:
        
    
    ############################################################################
    ############################################################################
    ############################  Execution: FFNN_w  ###########################
    ############################################################################
    ############################################################################
    
    preprocessing_units = {
        "SDF": {"w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()]},
        "RP": {"w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()]},
        "VP": {"w":[PPB_nan_to_0()]},
        "VP_HARQ": {"w":[PPB_nan_to_0()]}
        }
    
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-5)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    structures = {
        "SDF":{
            "input_dim":(41,1), "channels":1,
            "FFNN":[
                {"out_features":32},{"out_features":16}, {"out_features":8},
                {"out_features":4}, {"out_features":1}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.Softplus(), "dropout":0.05}]*4},
        "RP":{
            "input_dim":(41,1), "channels":1,
            "FFNN":[{"out_features":64}, {"out_features":32}, {"out_features":16}, 
                    {"out_features":8}, {"out_features":1}], 
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(0.1), "dropout":0.02}]*4}, 
        "VP":{
            "input_dim":(41,1), "channels":1,
            "FFNN":[{"out_features":128},{"out_features":64}, {"out_features":1}], ## exp!
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.Softsign(), "dropout":0.1}]*2}, 
        "VP_HARQ":{
            "input_dim":(6,1), "channels":1,
            "FFNN":[{"out_features":128}, {"out_features":64}, {"out_features":1}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.Softsign(), "dropout":0.0}]*2}}
    
    for objective_str in ["SDF","RP","VP","VP_HARQ"]:
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model[objective_str]["val_mode"]
    
    years_execution_ = list(years_execution)
    #  years_execution_ = range(2019,2023) # [2008,2020,] # 
    # train_validate_pars.update({"to_epoch":25})
    
    results = {
        "pred":{},
        "perf_pred":{},
        "perf_port":{},
        "Scheduler_Monitor":{},
        "pred_stats":{}
        }
    total_portfolio = {}
    timers = {}
    
    for year in years_execution_:
        # year = 2012
        timers[year] = {}
        for objective_str in ["SDF","RP","VP","VP_HARQ"]:
            # objective_str = "VP"
            
            ## objective related parameters
            if objective_str in ["RP","VP","VP_HARQ"]:
                objective_FFNN = RP_objective(
                    loss_fn = Loss_ENET,
                    loss_fn_par = {
                        "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                        "lambda_p"  : 0.0000001 , "volatility":True if "VP" in objective_str else False},
                    quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            else:
                objective_FFNN = SDF_objective(
                    loss_fn = Loss_SDF, loss_fn_par = {
                        "normalise":"1a", "return_mode":"simple",
                        "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            
            ## load target variable for frequency
            target_variable = TARGET_VARIABLES[objective_str]
    
            ## further objective specific settings: input columns and input dim
            if objective_str == "VP_HARQ":
                use_columns = HARQ_columns
                params_hyperparameter_structure["input_dim"] = (len(use_columns),1)
            else:
                use_columns = []
                params_hyperparameter_structure["input_dim"] = (41,1)
            
            
            ## dataset import
            params_FFNN_w_dataset = params_hyperparameter_dataset|{}
            params_FFNN_w_dataset.update(pars_overwrite_model[objective_str])
            params_FFNN_w_dataset["merge_freq"] = False
            params_FFNN_w_dataset["test_date"] = f"{year}-12-31"
            params_FFNN_w_dataset["test_len"] = 12
            params_FFNN_w_dataset.update(
                {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                                  'dataset': {'w': []}, 'filter': {'w': []}}})
            dataset_FFNN_w = PhD4_dataset(
                **params_FFNN_w_dataset,
                target_variable = target_variable,
                use_columns = use_columns)
            dataset_FFNN_w.shuffle()
            
            subname = "FFNN_w_"+objective_str
            
            # fc_objective = objective_FFNN ## to delete
                
            if subname not in results["pred"].keys():
                results["pred"][subname] = []
                results["perf_pred"][subname] = []
                results["perf_port"][subname] = []
                results["pred_stats"][subname] = []
            timers[year][subname] = {}
            
            print_hint2(f"{year}-12-31: {subname:12s}",2,2,width_field = 50)
            fc_naive_test = pd.DataFrame()
            
            year_results = {
                "prediction_results" : {},
                "portfolio" : pd.DataFrame(),
                "portfolio_results" : [],
                "prediction_stats":[]
                }
            
            params_model = params_hyperparameter_model|{}
            del params_model["model_type"]
            params_model["structure"] = Layers_FFNN(structures[objective_str])
            params_model.update(model_parameter_updates[objective_str])
            
            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
            
            ## update parameters for model
            params_model.update(pars_overwrite_model[objective_str])
            
            model_parameters = {
                **params_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_FFNN}
            
            jobs =[[f"FFNN_w_{objective_str}_{i}_{year}",copy.deepcopy(model_parameters)]\
                             for i in range(N_MODELS)]
            execution_jobs = []
            # process_group = TO_DDP_PG(rank=0) # see beginning of script
            for n_worker in range(N_PARALLEL):
                # n_worker = 0
                thread_arguments = {
                    "data_set":dataset_FFNN_w,
                    "model_type":FFNN,
                    "jobs":jobs,
                    "year_results":year_results,
                    "thread_nr":n_worker,
                    "timers":timers[year][subname],
                    "train_validate_pars":train_validate_pars,
                    "test_pars":test_pars,
                    "model_file_path":MODELS_PATH,
                    "verbose":1}
                thread = th.Thread(target=PhD2_year_job_thread,kwargs=thread_arguments)
                execution_jobs.append(thread)
            for j in execution_jobs:
                j.start()
            for j in execution_jobs:
                j.join()
            # del process_group
            print("Processes complete.")
            
            
            ####
            
            # prediction_results contains all results for predictive performance
            res_naive = fc_naive(year_results["portfolio"].copy(),pd.DataFrame(),
                                 objective_FFNN,fc_pars)
            
            year_results["prediction_results"]["res_naive"] = res_naive[0][0]
            res_naive[0][1]["model"] = "res_naive"
            year_results["portfolio_results"].append(res_naive[0][1])
            year_results["portfolio"]["res_naive"] = res_naive[0][2]["pred"]
            res_naive[0][3]["model"] = "res_naive"
            year_results["prediction_stats"].append(res_naive[0][3])
            
            results["pred"][subname].append(year_results["portfolio"])
            
            # showcase total portfolios of forecast combinations:
            if subname not in total_portfolio.keys():
                total_portfolio[subname] = {}
                total_portfolio[subname]["res_naive"] = res_naive[0][1]
            else:
                total_portfolio[subname]["res_naive"] = pd.concat([
                    total_portfolio[subname]["res_naive"], res_naive[0][1]])
            
            if objective_str in ["RP","SDF"]:
                tpf_dict = total_portfolio[subname]
                aggregate = []
                show_columns = ['sample_avg', 'HML', 'high', 'low']
                if objective_str == "SDF":
                    show_columns.append("SDF")
                for key in tpf_dict.keys():
                    aggregate.append((tpf_dict[key].loc[tpf_dict[key].index == "aggregate",
                                                   show_columns]+1).prod(axis=0).to_frame(name=key).T)
                print(pd.concat(aggregate))
            
            for subsubmodel in year_results["prediction_results"].keys():
                year_results["prediction_results"][subsubmodel]["HMLQAD"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][3]
                year_results["prediction_results"][subsubmodel]["HMLQAA"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][2]
                year_results["prediction_results"][subsubmodel]["QAD"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][1]
                year_results["prediction_results"][subsubmodel]["QAA"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][0]
            year_results["prediction_results"] = pd.DataFrame(year_results["prediction_results"])
            year_results["prediction_results"]["year"] = year
            results["perf_pred"][subname].append(year_results["prediction_results"])
            
            if objective_str in ["RP","SDF"]:
                year_results["portfolio_results"] = pd.concat(year_results["portfolio_results"]).reset_index()
                year_results["portfolio_results"].loc[
                    year_results["portfolio_results"]["date"]=="aggregate",
                    "date"] = year
                year_results["portfolio_results"].set_index("date",inplace=True)
                results["perf_port"][subname].append(year_results["portfolio_results"])
            else:
                year_results["portfolio_results"] = pd.DataFrame()
                results["perf_port"][subname] = [pd.DataFrame()]
                
            
            year_results["prediction_stats"] = pd.concat(year_results["prediction_stats"]).reset_index()
            year_results["prediction_stats"].loc[
                year_results["prediction_stats"]["date"]=="aggregate",
                "date"] = year
            year_results["prediction_stats"].set_index("date",inplace=True)
            results["pred_stats"][subname].append(year_results["prediction_stats"])
            
            year_results["prediction_results"].\
                to_hdf(RESULTS_PATH+ f"FFNN_w_{objective_str}_{variant}_perf_pred"+\
                       f"{year}{test_appendix}.h5","data")
            year_results["portfolio_results"].\
                to_hdf(RESULTS_PATH+ f"FFNN_w_{objective_str}_{variant}_perf_port"+\
                              f"{year}{test_appendix}.h5","data")
            year_results["portfolio"].\
                to_hdf(RESULTS_PATH+ f"FFNN_w_{objective_str}_{variant}_pred"+\
                              f"{year}{test_appendix}.h5","data")
            year_results["prediction_stats"].\
                to_hdf(RESULTS_PATH+ f"FFNN_w_{objective_str}_{variant}_pred_stats"+\
                              f"{year}{test_appendix}.h5","data")
            with open(TIMER_PATH+f"FFNN_w_{objective_str}_{variant}_pred_stats"+\
                          f"{year}{test_appendix}.txt", 'w') as f:
                f.write(json.dumps(timers[year]))
    
    
    ############################################################################
    ############################################################################
    ########################  Execution: FFNN_wmq_MIFDEL  ######################
    ############################################################################
    ############################################################################
    
    ## parameters from previous stages
    preprocessing_units = {
        "SDF": {
            "w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]},
        "RP": {
            "w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()],
            "m":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=366,span=2), PPB_nan_to_0()]},
        "VP": {
            "w":[PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}}
    
    model_parameter_updates = {
        "SDF": {"learning_rate": 0.0009118819655545162},
        "RP": {"learning_rate": 0.006737946999085467},
        "VP": {"learning_rate": 0.006737946999085467},
        "VP_HARQ": {"learning_rate": 0.006737946999085467},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    structures_w = {
        "SDF":{
            "FFNN":[{"out_features":64},{"out_features":32}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2}, 
        "RP":{
            "FFNN":[{"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":act_DoubleTanh(), "dropout":0.05}]*1},
        "VP":{
            "FFNN":[{"out_features":32}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.Softsign(), "dropout":0.05}]*1}} 
    
    
    structures_m = {
        "SDF":{
            "FFNN":[{"out_features":32},{"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2},
        "RP":{
            "FFNN":[{"out_features":32},{"out_features":16},{"out_features":8}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*3},
        "VP":{
            "FFNN":[{"out_features":64},{"out_features":32}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2}} 
    
    structures_q = {
        "SDF":{
            "FFNN":[{"out_features":32},{"out_features":16}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.Softsign(), "dropout":0.05}]*2},
        "RP":{
            "FFNN":[{"out_features":64}, {"out_features":32}, {"out_features":16}, {"out_features":8}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*4},
        "VP":{
            "FFNN":[{"out_features":16}, {"out_features":8}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*2}}
    
    strutcutes_joiner = {
        "SDF":{
            "FFNN":[{"out_features":64},{"out_features":32},{"out_features":16},{"out_features":1}],
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*3},
        "RP":{
            "FFNN":[{"out_features":32},{"out_features":1}],
            "FFNN_sec":[{"activation":nn.Hardtanh(), "dropout":0.05}]*1},
        "VP":{
            "FFNN":[{"out_features":16},{"out_features":1}],
            "FFNN_sec":[{"activation":nn.LeakyReLU(), "dropout":0.05}]*1},
            }
    
    structure_default =  {
        "kernel_order": ["w","m","q"],
        "kernels": {
            "w":{"input_dim":(41,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                 "FFNN":[{"out_features":32}, {"out_features":16}],
                 "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                 "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                              {"activation":nn.ReLU(), "dropout":0.02},],
                 "TNN":[], "reshape":[True,True], "postprocessor":None},
             "m":{"input_dim":(26,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                  "FFNN":[{"out_features":32}, {"out_features":16}],
                  "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                  "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                               {"activation":nn.ReLU(), "dropout":0.02},],
                  "TNN":[], "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay}},
             "q":{"input_dim":(76,1),"in_channels":1, "CNN":[], "CNN_sec": [],
                  "FFNN":[{"out_features":32}, {"out_features":16}],
                  "CNN_norm":BatchNorm_self, "FFNN_norm":nn.BatchNorm1d,
                  "FFNN_sec": [{"activation":nn.ReLU(), "dropout":0.02},
                               {"activation":nn.ReLU(), "dropout":0.02},],
                  "TNN":[], "reshape":[True,True], "postprocessor":{"layer_type":Layer_Sigmoid_decay}}},
    "FFNN":[
        {"out_features":24}, {"out_features":1},
            ],
    "FFNN_sec":[{"activation":nn.ReLU(), "dropout":0.02},
                {"activation":nn.ReLU(), "dropout":0.02}]}
    
    for objective_str in ["SDF","RP","VP"]:
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model[objective_str]["val_mode"]
    
    years_execution_ = list(years_execution)
    # years_execution_ = range(2016,2023)
    
    results = {
        "pred":{},
        "perf_pred":{},
        "perf_port":{},
        "Scheduler_Monitor":{},
        "pred_stats":{}
        }
    total_portfolio = {}
    timers = {}
    
    params_hyperparameter_dataset.update({
        "input_frequencies": ["w","m","q"],
        "lookback":{"w":1,"m":1,"q":1}})
    
    for year in years_execution_:
        # year = 2003
        timers[year] = {}
        for objective_str in ["SDF","RP","VP"]:
            # objective_str = "VP"
            
            ## objective related parameters
            if objective_str in ["RP","VP","VP_HARQ"]:
                objective_FFNN = RP_objective(
                    loss_fn = Loss_ENET,
                    loss_fn_par = {
                        "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                        "lambda_p"  : 0.0000001 , "volatility":True if "VP" in objective_str else False},
                    quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            else:
                objective_FFNN = SDF_objective(
                    loss_fn = Loss_SDF, loss_fn_par = {
                        "normalise":"1a", "return_mode":"simple",
                        "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            
            ## load target variable for frequency
            target_variable = TARGET_VARIABLES[objective_str]
            
            ## dataset import
            params_FFNN_wmq_dataset = params_hyperparameter_dataset|{}
            params_FFNN_wmq_dataset.update(pars_overwrite_model[objective_str])
            params_FFNN_wmq_dataset["merge_freq"] = False
            params_FFNN_wmq_dataset["test_date"] = f"{year}-12-31"
            params_FFNN_wmq_dataset["test_len"] = 12
            
            params_FFNN_wmq_dataset.update(
                {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                                  'dataset': {'w': [],"m":[],"q":[]}, 
                                  'filter': {'w': [],"m":[],"q":[]}}})
            params_FFNN_wmq_dataset["meta_series"] = {"m":"m_dist","q":"q_dist"}
            
            dataset_FFNN_wmq = PhD4_dataset(
                **params_FFNN_wmq_dataset,
                target_variable = target_variable,
                use_columns = [])
            dataset_FFNN_wmq.shuffle()
            
            subname = "FFNN_w_"+objective_str
            
            # fc_objective = objective_FFNN ## to delete
                
            if subname not in results["pred"].keys():
                results["pred"][subname] = []
                results["perf_pred"][subname] = []
                results["perf_port"][subname] = []
                results["pred_stats"][subname] = []
            timers[year][subname] = {}
            
            print_hint2(f"{year}-12-31: {subname:12s}",2,2,width_field = 50)
            fc_naive_test = pd.DataFrame()
            
            year_results = {
                "prediction_results" : {},
                "portfolio" : pd.DataFrame(),
                "portfolio_results" : [],
                "prediction_stats":[]
                }
            
            local_structure = copy.deepcopy(structure_default)
            local_structure["kernels"]["w"].update(structures_w[objective_str])
            local_structure["kernels"]["m"].update(structures_m[objective_str])
            local_structure["kernels"]["q"].update(structures_q[objective_str])
            local_structure.update(strutcutes_joiner[objective_str])
            
            params_model = params_hyperparameter_model|{}
            params_model.update(model_parameter_updates[objective_str])
            params_model["structure"] = Layers_multifreq(local_structure)
            del params_model["model_type"]
            
            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
            
            ## update parameters for model
            params_model.update(pars_overwrite_model[objective_str])
            
            model_parameters = {
                **params_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_FFNN}
            
            jobs =[[f"FFNN_wmq_{objective_str}_{i}_{year}",copy.deepcopy(model_parameters)]\
                             for i in range(N_MODELS)]
            execution_jobs = []
            # process_group = TO_DDP_PG(rank=0) # see beginning of script
            for n_worker in range(N_PARALLEL):
                # n_worker = 0
                thread_arguments = {
                    "data_set":dataset_FFNN_wmq,
                    "model_type":ANN_MultiFreq,
                    "jobs":jobs,
                    "year_results":year_results,
                    "thread_nr":n_worker,
                    "timers":timers[year][subname],
                    "train_validate_pars":train_validate_pars,
                    "test_pars":test_pars,
                    "model_file_path":MODELS_PATH,
                    "verbose":1}
                thread = th.Thread(target=PhD2_year_job_thread,kwargs=thread_arguments)
                execution_jobs.append(thread)
            for j in execution_jobs:
                j.start()
            for j in execution_jobs:
                j.join()
            # del process_group
            print("Processes complete.")
            
            # prediction_results contains all results for predictive performance
            res_naive = fc_naive(year_results["portfolio"].copy(),pd.DataFrame(),
                                 objective_FFNN,fc_pars)
            
            year_results["prediction_results"]["res_naive"] = res_naive[0][0]
            res_naive[0][1]["model"] = "res_naive"
            year_results["portfolio_results"].append(res_naive[0][1])
            year_results["portfolio"]["res_naive"] = res_naive[0][2]["pred"]
            res_naive[0][3]["model"] = "res_naive"
            year_results["prediction_stats"].append(res_naive[0][3])
            
            results["pred"][subname].append(year_results["portfolio"])
            
            # showcase total portfolios of forecast combinations:
            if subname not in total_portfolio.keys():
                total_portfolio[subname] = {}
                total_portfolio[subname]["res_naive"] = res_naive[0][1]
            else:
                total_portfolio[subname]["res_naive"] = pd.concat([
                    total_portfolio[subname]["res_naive"], res_naive[0][1]])
            
            if objective_str in ["RP","SDF"]:
                tpf_dict = total_portfolio[subname]
                aggregate = []
                show_columns = ['sample_avg', 'HML', 'high', 'low']
                if objective_str == "SDF":
                    show_columns.append("SDF")
                for key in tpf_dict.keys():
                    aggregate.append((tpf_dict[key].loc[tpf_dict[key].index == "aggregate",
                                                   show_columns]+1).prod(axis=0).to_frame(name=key).T)
                print(pd.concat(aggregate))
            
            for subsubmodel in year_results["prediction_results"].keys():
                year_results["prediction_results"][subsubmodel]["HMLQAD"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][3]
                year_results["prediction_results"][subsubmodel]["HMLQAA"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][2]
                year_results["prediction_results"][subsubmodel]["QAD"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][1]
                year_results["prediction_results"][subsubmodel]["QAA"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][0]
            year_results["prediction_results"] = pd.DataFrame(year_results["prediction_results"])
            year_results["prediction_results"]["year"] = year
            results["perf_pred"][subname].append(year_results["prediction_results"])
            
            if objective_str in ["RP","SDF"]:
                year_results["portfolio_results"] = pd.concat(year_results["portfolio_results"]).reset_index()
                year_results["portfolio_results"].loc[
                    year_results["portfolio_results"]["date"]=="aggregate",
                    "date"] = year
                year_results["portfolio_results"].set_index("date",inplace=True)
                results["perf_port"][subname].append(year_results["portfolio_results"])
            else:
                year_results["portfolio_results"] = pd.DataFrame()
                results["perf_port"][subname] = [pd.DataFrame()]
                
            
            year_results["prediction_stats"] = pd.concat(year_results["prediction_stats"]).reset_index()
            year_results["prediction_stats"].loc[
                year_results["prediction_stats"]["date"]=="aggregate",
                "date"] = year
            year_results["prediction_stats"].set_index("date",inplace=True)
            results["pred_stats"][subname].append(year_results["prediction_stats"])
            
            year_results["prediction_results"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_mixed_{objective_str}_{variant}_perf_pred"+\
                       f"{year}{test_appendix}.h5","data")
            year_results["portfolio_results"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_mixed_{objective_str}_{variant}_perf_port"+\
                              f"{year}{test_appendix}.h5","data")
            year_results["portfolio"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_mixed_{objective_str}_{variant}_pred"+\
                              f"{year}{test_appendix}.h5","data")
            year_results["prediction_stats"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_mixed_{objective_str}_{variant}_pred_stats"+\
                              f"{year}{test_appendix}.h5","data")
            with open(TIMER_PATH+f"FFNN_wmq_mixed_{objective_str}_{variant}_pred_stats"+\
                          f"{year}{test_appendix}.txt", 'w') as f:
                f.write(json.dumps(timers[year]))
    
    
    
    ############################################################################
    ############################################################################
    #######################  Execution: FFNN_wmq_joined  #######################
    ############################################################################
    ############################################################################
    
    
    preprocessing_units = {
        "SDF": {
            "w":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=92,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]},
        "RP": {
            "w":[PPB_norm_df(rolling=93,span=2), PPB_nan_to_0()],
            "m":[PPB_norm_df(rolling=0,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=366,span=2), PPB_nan_to_0()]},
        "VP": {
            "w":[PPB_nan_to_0()],
            "m":[PPB_rank_df(rolling=183,span=2), PPB_nan_to_0()],
            "q":[PPB_norm_df(rolling=92,span=2), PPB_nan_to_0()]}}
    
    model_parameter_updates = {
        "SDF": {"learning_rate": np.exp(-5)},
        "RP": {"learning_rate": np.exp(-5)},
        "VP": {"learning_rate": np.exp(-5)},
        "VP_HARQ": {"learning_rate": np.exp(-5)},
        }
    
    model_parameter_base_penalties = {
        "SDF": [ OBJ_P_long_short_ratio()], 
        "RP": Loss_Huber_weighted(threshold = 0.01),
        "VP": Loss_Huber_weighted(threshold = 0.1),
        "VP_HARQ": Loss_Huber_weighted(threshold = 0.01),
        }
    
    structures = {
        "SDF":{
            "input_dim":(143,1), "channels":1,
            "FFNN":[
                {"out_features":32},{"out_features":16}, {"out_features":8}, {"out_features":1}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(.1), "dropout":0.05}]*4},
        "RP":{
            "input_dim":(143,1), "channels":1,
            "FFNN":[{"out_features":32}, {"out_features":16}, 
                    {"out_features":1}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.LeakyReLU(0.1), "dropout":0.00}]*4}, 
        "VP":{
            "input_dim":(143,1), "channels":1,
            "FFNN":[{"out_features":64}, {"out_features":32}, {"out_features":16}, {"out_features":1}],
            "FFNN_norm":nn.BatchNorm1d,
            "FFNN_sec":[{"activation":nn.Tanh(), "dropout":0.0}]*3}}
    
    
    for objective_str in ["SDF","RP","VP"]:
        with open(HYPERPARAMETER_PATH+f"{objective_str}_{variant}_model_hyp2.txt", 'r') as readfile:
            pars_overwrite_model[objective_str] = json.load(readfile)
        pars_overwrite_model[objective_str]["mode_validation"] = pars_overwrite_model[objective_str]["val_mode"]
    
    years_execution_ = list(years_execution)
    # years_execution_ = range(2004,2016)
    
    results = {
        "pred":{},
        "perf_pred":{},
        "perf_port":{},
        "Scheduler_Monitor":{},
        "pred_stats":{}
        }
    total_portfolio = {}
    timers = {}
    
    params_hyperparameter_dataset.update({
        "input_frequencies": ["w","m","q"],
        "lookback":{"w":1,"m":1,"q":1}})
    
    for year in years_execution_:
        # year = 2003
        timers[year] = {}
        for objective_str in ["SDF","RP","VP"]:
            # objective_str = "RP"
            
            ## objective related parameters
            if objective_str in ["RP","VP","VP_HARQ"]:
                objective_FFNN = RP_objective(
                    loss_fn = Loss_ENET,
                    loss_fn_par = {
                        "base_penalty": model_parameter_base_penalties[objective_str], "alpha_p"   : 0.00001, 
                        "lambda_p"  : 0.0000001 , "volatility":True if "VP" in objective_str else False},
                    quantiles = 10, target_is_return=False if "VP" in objective_str else True)
            else:
                objective_FFNN = SDF_objective(
                    loss_fn = Loss_SDF, loss_fn_par = {
                        "normalise":"1a", "return_mode":"simple",
                        "mode":"SDF", "penalties":model_parameter_base_penalties[objective_str]})
            
            ## load target variable for frequency
            target_variable = TARGET_VARIABLES[objective_str]
    
            ## further objective specific settings: input columns and input dim
            if objective_str == "VP_HARQ":
                use_columns = HARQ_columns
                params_hyperparameter_structure["input_dim"] = (len(use_columns),1)
            else:
                use_columns = []
                params_hyperparameter_structure["input_dim"] = (41,1)
            
            
            ## dataset import
            params_FFNN_wmq_joined_dataset = params_hyperparameter_dataset|{}
            params_FFNN_wmq_joined_dataset.update(pars_overwrite_model[objective_str])
            params_FFNN_wmq_joined_dataset["merge_freq"] = True
            params_FFNN_wmq_joined_dataset["test_date"] = f"{year}-12-31"
            params_FFNN_wmq_joined_dataset["test_len"] = 12
            params_FFNN_wmq_joined_dataset.update(
                {"preprocessing":{'batchwise': preprocessing_units[objective_str], 
                                  'dataset': {'w': [], 'm':[],'q':[]}, 
                                  'filter': {'w': [], 'm':[],'q':[]}}})
            dataset_FFNN_w = PhD4_dataset(
                **params_FFNN_wmq_joined_dataset,
                target_variable = target_variable,
                use_columns = use_columns)
            dataset_FFNN_w.shuffle()
            
            subname = "FFNN_wmq_joined_"+objective_str
            
            # fc_objective = objective_FFNN ## to delete
                
            if subname not in results["pred"].keys():
                results["pred"][subname] = []
                results["perf_pred"][subname] = []
                results["perf_port"][subname] = []
                results["pred_stats"][subname] = []
            timers[year][subname] = {}
            
            print_hint2(f"{year}-12-31: {subname:12s}",2,2,width_field = 50)
            fc_naive_test = pd.DataFrame()
            
            year_results = {
                "prediction_results" : {},
                "portfolio" : pd.DataFrame(),
                "portfolio_results" : [],
                "prediction_stats":[]
                }
            
            params_model = params_hyperparameter_model|{}
            del params_model["model_type"]
            params_model["structure"] = Layers_FFNN(structures[objective_str])
            params_model.update(model_parameter_updates[objective_str])
            
            scheduler_monitor = Scheduler_Monitor(**params_hyperparameter_scheduler_monitor)
            
            ## update parameters for model
            params_model.update(pars_overwrite_model[objective_str])
            
            model_parameters = {
                **params_model,
                "Scheduler_Monitor": scheduler_monitor,
                "objective": objective_FFNN}
            
            jobs =[[f"FFNN_wmq_joined_{objective_str}_{i}_{year}",copy.deepcopy(model_parameters)]\
                             for i in range(N_MODELS)]
            execution_jobs = []
            # process_group = TO_DDP_PG(rank=0) # see beginning of script
            for n_worker in range(N_PARALLEL):
                # n_worker = 0
                thread_arguments = {
                    "data_set":dataset_FFNN_w,
                    "model_type":FFNN,
                    "jobs":jobs,
                    "year_results":year_results,
                    "thread_nr":n_worker,
                    "timers":timers[year][subname],
                    "train_validate_pars":train_validate_pars,
                    "test_pars":test_pars,
                    "model_file_path":MODELS_PATH,
                    "verbose":1}
                thread = th.Thread(target=PhD2_year_job_thread,kwargs=thread_arguments)
                execution_jobs.append(thread)
            for j in execution_jobs:
                j.start()
            for j in execution_jobs:
                j.join()
            # del process_group
            print("Processes complete.")
            
            
            ####
            
            # prediction_results contains all results for predictive performance
            res_naive = fc_naive(year_results["portfolio"].copy(),pd.DataFrame(),
                                 objective_FFNN,fc_pars)
            
            year_results["prediction_results"]["res_naive"] = res_naive[0][0]
            res_naive[0][1]["model"] = "res_naive"
            year_results["portfolio_results"].append(res_naive[0][1])
            year_results["portfolio"]["res_naive"] = res_naive[0][2]["pred"]
            res_naive[0][3]["model"] = "res_naive"
            year_results["prediction_stats"].append(res_naive[0][3])
            
            results["pred"][subname].append(year_results["portfolio"])
            
            # showcase total portfolios of forecast combinations:
            if subname not in total_portfolio.keys():
                total_portfolio[subname] = {}
                total_portfolio[subname]["res_naive"] = res_naive[0][1]
            else:
                total_portfolio[subname]["res_naive"] = pd.concat([
                    total_portfolio[subname]["res_naive"], res_naive[0][1]])
            
            if objective_str in ["RP","SDF"]:
                tpf_dict = total_portfolio[subname]
                aggregate = []
                show_columns = ['sample_avg', 'HML', 'high', 'low']
                if objective_str == "SDF":
                    show_columns.append("SDF")
                for key in tpf_dict.keys():
                    aggregate.append((tpf_dict[key].loc[tpf_dict[key].index == "aggregate",
                                                   show_columns]+1).prod(axis=0).to_frame(name=key).T)
                print(pd.concat(aggregate))
            
            for subsubmodel in year_results["prediction_results"].keys():
                year_results["prediction_results"][subsubmodel]["HMLQAD"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][3]
                year_results["prediction_results"][subsubmodel]["HMLQAA"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][2]
                year_results["prediction_results"][subsubmodel]["QAD"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][1]
                year_results["prediction_results"][subsubmodel]["QAA"] = \
                    year_results["prediction_results"][subsubmodel]["QAA"][0]
            year_results["prediction_results"] = pd.DataFrame(year_results["prediction_results"])
            year_results["prediction_results"]["year"] = year
            results["perf_pred"][subname].append(year_results["prediction_results"])
            
            if objective_str in ["RP","SDF"]:
                year_results["portfolio_results"] = pd.concat(year_results["portfolio_results"]).reset_index()
                year_results["portfolio_results"].loc[
                    year_results["portfolio_results"]["date"]=="aggregate",
                    "date"] = year
                year_results["portfolio_results"].set_index("date",inplace=True)
                results["perf_port"][subname].append(year_results["portfolio_results"])
            else:
                year_results["portfolio_results"] = pd.DataFrame()
                results["perf_port"][subname] = [pd.DataFrame()]
                
            
            year_results["prediction_stats"] = pd.concat(year_results["prediction_stats"]).reset_index()
            year_results["prediction_stats"].loc[
                year_results["prediction_stats"]["date"]=="aggregate",
                "date"] = year
            year_results["prediction_stats"].set_index("date",inplace=True)
            results["pred_stats"][subname].append(year_results["prediction_stats"])
            
            year_results["prediction_results"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_joined_{objective_str}_{variant}_perf_pred"+\
                       f"{year}{test_appendix}.h5","data")
            year_results["portfolio_results"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_joined_{objective_str}_{variant}_perf_port"+\
                              f"{year}{test_appendix}.h5","data")
            year_results["portfolio"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_joined_{objective_str}_{variant}_pred"+\
                              f"{year}{test_appendix}.h5","data")
            year_results["prediction_stats"].\
                to_hdf(RESULTS_PATH+ f"FFNN_wmq_joined_{objective_str}_{variant}_pred_stats"+\
                              f"{year}{test_appendix}.h5","data")
            with open(TIMER_PATH+f"FFNN_wmq_joined_{objective_str}_{variant}_pred_stats"+\
                          f"{year}{test_appendix}.txt", 'w') as f:
                f.write(json.dumps(timers[year]))
    



