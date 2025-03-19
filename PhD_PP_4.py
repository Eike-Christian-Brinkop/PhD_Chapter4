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


This script is for the calculation of the 143 key figures used in this Chapter.
- replace BASE_PATH with the path on your machine.
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


####################################################################################################
####################################################################################################
############################################  Packages  ############################################
####################################################################################################
####################################################################################################

import numpy as np, pandas as pd, datetime as dt, os
from BT_utils import print_fmt, day_shift_nodt, month_shift_num, ELK_progress_bar
from pandas.tseries.offsets import MonthEnd
from matplotlib import pyplot as plt
# import wrds

import numba as nb, warnings
from numba.core.errors import NumbaDeprecationWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# from alive_progress import alive_it, alive_bar

PATH_DATA="D:/Backup Research/Data_P3/"

'''
mode guide:
    "FULL": do all
    
    "mergeing": only rerun mergeing of CRSP and Compustat
    "quarterly": only do quarterly kf excluding daily mcap recalculation
    "mcap": only run daily mcap
    "weekly": only run weekly
    "monthly": run monthly kf calculation
    "ind": run quarterly industry
    "linkeage": run linkeage of frequencies
    "partitioning": run partitioning of data
    
'''
mode = ["False"]
min_companies_per_industry_month = 5

####################################################################################################
####################################################################################################
###################################  Merging CRSP and Compustat  ###################################
####################################################################################################
####################################################################################################
'''
--- Section description ---
In this section of the code, we match compustat and crsp identifiers by their cusip code.
Alternatively, you can also match using the official matching table. 
Summary Statistics for the matching are produced and presented underneath this section.
'''


if ("FULL" in mode or "mergeing" in mode):
    
    #### Compustat
    ## read compustat quarterly file
    comp_q = pd.read_csv(
        PATH_DATA+"quarterly.csv",usecols=["gvkey","cusip","sic"])
    comp_q.dropna(inplace=True)
    comp_ident_q = comp_q.drop_duplicates(subset = ["gvkey","cusip"])
    ## take only the first 8 characters of cusip codes:
    comp_ident_q.loc[:,"cusip"] = comp_ident_q["cusip"].str[:8]
    
    #### CRSP
    crsp_m = pd.read_csv(
        PATH_DATA+"monthly.txt",usecols=["PERMNO","CUSIP","SICCD"],sep="\t")
    crsp_m.dropna(inplace=True)
    crsp_ident_m = crsp_m.drop_duplicates(subset = ["PERMNO","CUSIP"])
    
    
    ## merge the two, outer join to run some summary statistics
    comp_ident_joined = pd.merge(
        comp_ident_q,crsp_ident_m,
        left_on = "cusip",right_on = "CUSIP",how="outer")
    ## summary statistics:
    shape = comp_ident_joined.shape[0]
    matched = comp_ident_joined[comp_ident_joined.isna().sum(axis=1)==0].shape[0]
    compustat_not_crsp = comp_ident_joined[comp_ident_joined["PERMNO"].isna()]
    crsp_not_compustat = comp_ident_joined[comp_ident_joined["gvkey"].isna()]
    print_fmt("Summary Statistics for matching Compustat and CRSP",0,3,75)
    print(f"\nThere are   {shape:d}  total unique identifiers by cusip codes in both Compustat and CRSP.", 
          f"\nOf those,  {matched:d}  are matched between compustat and CRSP",
          f"\n{compustat_not_crsp.shape[0]:d}  identifiers exist only in Compustat, while",
          f"\n{crsp_not_compustat.shape[0]:d}  identifiers exist only in CRSP\n")
    
    crsp_shape_matched = crsp_m[crsp_m["PERMNO"].isin(comp_ident_joined.dropna()["PERMNO"])].shape[0]
    crsp_shape_unmatched = crsp_m.shape[0]-crsp_shape_matched
    
    
    comp_shape_matched = comp_q[comp_q["gvkey"].isin(comp_ident_joined.dropna()["gvkey"])].shape[0]
    comp_shape_unmatched = comp_q.shape[0]-comp_shape_matched
    
    print(f"\nMatched CRSP     : {crsp_shape_matched:8d}; Unmatched CRSP     : {crsp_shape_unmatched:8d}",
          f"% matched: {crsp_shape_matched/(crsp_shape_unmatched+crsp_shape_matched):.2%}"
          f"\nMatched Compustat: {comp_shape_matched:8d}; Unmatched Compustat: {comp_shape_unmatched:8d}",
          f"% matched: {comp_shape_matched/(comp_shape_unmatched+comp_shape_matched):.2%}")
    
    matching_table = comp_ident_joined.dropna()
    matching_table["gvkey"] = matching_table["gvkey"].astype(int)
    matching_table["PERMNO"] = matching_table["PERMNO"].astype(int)
    
    
    matching_table.to_hdf(PATH_DATA+"CRSP_Compustat_match.h5",key="data")
    
    del matching_table, comp_ident_joined, crsp_m, comp_q, crsp_ident_m, comp_ident_q

'''
##########   Summary Statistics for matching Compustat and CRSP   #########

There are   45051  total unique identifiers by cusip codes in both Compustat and CRSP. 
Of those,  27892  are matched between compustat and CRSP 
8525  identifiers exist only in Compustat, while 
8634  identifiers exist only in CRSP


Matched CRSP     :  3886925; Unmatched CRSP     :   736925 % matched: 84.06%
Matched Compustat:  1409057; Unmatched Compustat:   335967 % matched: 80.75%

'''
####################################################################################################
####################################################################################################
################################  Quarterly (from Compustat) [1/2]  ################################
####################################################################################################
####################################################################################################

def delta_series(column,shift=1,identifier = "gvkey",replace=0):
    column = (column-column.groupby(identifier).shift(shift))
    if replace is not None:
        column = column.replace(np.nan,replace)
    return column

def delta_series_relative(column,shift=1,identifier="gvkey",replace=0):
    column_shift = column.groupby(identifier).shift(shift)
    delta = abs_dif_ratio(column,column_shift).replace([np.inf,-np.inf],replace)
    return delta

def abs_dif_ratio(a, b):
    return (a-b)/a.abs()

def abs_ratio(a,b):
    return(a/b.abs())

def rolling_series(column,rolling=4,identifier="gvkey",replace=0):
    column = column.groupby(identifier).apply(lambda x: x.rolling(rolling,1).\
                                              mean()*rolling).droplevel(0)
    if replace is not None:
        column = column.replace(np.nan,replace)
    return column

def windsorise(column,cut=.01,rolling_quarter=True,date_entity = "date",
               identifier = "gvkey",replace=None):
    '''
    --- Provisional Docstring ---
    Windsorising function for quarterly and monthly data.

    Parameters
    ----------
    column : pandas series or dataframe
        Pandas series or dataframe to windsorise.
    cut : float, DEFAULT 0.01
        Ratio of how much to cut off the edges.
    rolling_quarter : bool, DEFAULT True
        Wether to perform the windsorising on quarterly rolling windows.
    date_entity : str, DEFAULT "date"
        Date entity of the series.
    identifier : str, DEFAULT "gvkey"
        Firm level identifier.
    replace : float or None DEFAULT 0
        Wether to replace na values wíth a specific value.

    Returns
    -------
    column : pandas series
        Windsorised column.
        
    --------
    Testing:
    --------
        column = quarterly["acc"]; 
        cut = .01; 
        rolling_quarter = True
        date_entity = "date"; 
        identifier = 'gvkey'; 
        replace= None

    '''
    # 
    ## rolling window clipping needed.
    if rolling_quarter:
        rolling = 91
        if type(column) is not pd.DataFrame:
            column = column.to_frame(name = "series")
        column.sort_index(level = [date_entity,identifier],inplace=True)
        if "date_int" not in column.columns:
            column["date_int"] = column.index.get_level_values(date_entity)
            column["date_int"] = column["date_int"].apply(lambda x: str(int(x)))
            column["date_int"] = column["date_int"].apply(lambda x: dt.datetime.strptime(x,"%Y%m%d"))
            date_0 = column["date_int"].max()
            column["date_int"] = (column["date_int"]-date_0).dt.days
        column["date_int"]-=column["date_int"].values[0]
        dates = column["date_int"].to_numpy()
        del column["date_int"]
        
        x_index = column.index
        x_values = column.to_numpy()
        x_columns = column.columns
        del column
        x_norm_processed = np.ndarray(shape = x_values.shape)
        max_index,min_index = -1, 0
        while max_index+1 != len(dates):
            # 
            index_finder = np.where(dates==dates[max_index+1])[0]
            max_index = index_finder[-1]
            int_index = index_finder[0]
            min_index = np.where(dates>=dates[max_index]-rolling)[0][0]
            borders = (np.nanquantile(x_values[min_index:max_index+1,:],cut/2),
                       np.nanquantile(x_values[min_index:max_index+1,:],1-cut/2))
            local_array = np.clip(x_values[min_index:max_index+1,:],*borders)
            x_norm_processed[int_index:max_index+1,:] = local_array[int_index-min_index:,:]
        column = pd.DataFrame(x_norm_processed, index = x_index,columns = x_columns)
    else:
        limit = pd.DataFrame()
        limit["upper"] = column.groupby(date_entity).quantile(1-cut/2)
        limit["lower"] = column.groupby(date_entity).quantile(cut/2)
        column = pd.merge(column.to_frame(name="series"),limit,
                          left_on=date_entity,right_on=date_entity,how="inner")
        column["series"] = column[["series","lower"]].max(axis=1)
        column["series"] = column[["series","upper"]].min(axis=1)
    if replace is not None:
        column["series"] = column["series"].replace(np.nan,replace)
    return column.sort_index()

def column_stats(column):
    '''
    Describes pandas dataframe column with ratio of NA values, distribution stats, JB test,
    quantiles

    Parameters
    ----------
    column : pandas dataframe column
        DESCRIPTION.

    Returns
    -------
    List of summary stat values.
    
    --------
    Testing:
    --------
        column = quarterly["saleq"]
    
    '''
    if type(column) is pd.DataFrame:
        column = column[column.columns[0]]
    print_fmt(column.name,1,1,25)
    shape = column.shape[0]
    skew = column.skew(); kurt = column.kurtosis();
    analytics = {
        "NA" : column.isna().sum()/shape,
        "ZERO" : (column == 0).sum()/shape,
        "mean" : column.mean(), "std" : column.std(),
        "skew" : skew, "kurt" : kurt,
        "JB" : shape/6*(skew**2+(kurt-3)/4),
        "99.5%":column.quantile(.995), "0.5%":column.quantile(.005), 
        "min":column.min(),"max":column.max(),"median":column.median()}
    print(pd.DataFrame([analytics],index = [column.name]).round(4).T[:"kurt"].T,
          pd.DataFrame([analytics],index = [column.name]).round(4).T["JB":].T,sep="\n",end="\n\n\n")
    print(*analytics.values(), sep= ", ")
    return analytics

'''
--- Section description ---

'''
if ("FULL" in mode or "quarterly" in mode):
    #######################
    #  Data Preparations  #
    #######################
    #### loading, na processing, and merging with other table
    print_fmt("—"*25,1,0,75,sign="—")
    print_fmt("Preparing quarterly data",0,3,75)
    quarterly = pd.read_csv(PATH_DATA+"quarterly.csv", delimiter = ",")
    ## drop non-matches with CRSP file:
    shape_before = quarterly.shape[0]
    matching_table = pd.read_hdf(PATH_DATA+"CRSP_Compustat_match.h5","data")
    quarterly = pd.merge(quarterly,matching_table[["gvkey"]], left_on = "gvkey",right_on="gvkey",
                         how="inner")
    print(f"{shape_before-quarterly.shape[0]:8d} rows dropped, not matched with CRSP.",
          f"{quarterly.shape[0]:d} rows left.")
    
    print_fmt("—"*25,0,0,75,sign="—")
    ## drop duplicates on gvkey and datadate. Rules: sort by number of na values in row, 
    ## and drop the higher na values
    shape_before = quarterly.shape[0]
    quarterly["NA_row"] = quarterly.isna().sum(axis=1)
    quarterly.sort_values(by=["gvkey","datadate","NA_row","datacqtr"],ascending=True,inplace=True)
    quarterly.drop_duplicates(subset = ["gvkey","datadate"],keep="first",inplace=True)
    del quarterly["NA_row"]
    print(f"{shape_before-quarterly.shape[0]:8d} rows dropped, duplicate gvkey and datadate.",
          f"{quarterly.shape[0]:d} rows left.")
    
    
    '''
    ##########   Summary Statistics for Compustat quarterly observations   #########
    
    Matched Compustat:  1409057; Unmatched Compustat:   336354     % matched: 80.73%
    
    ———————————————————————————————————————————————————————————————————————————
    
         927 rows dropped, duplicate gvkey and datadate. 1461571 rows left.
    '''
    #### report dates and date handling
    ## date is either the report date or the lagged datadate
    ## we shift at least 20 days from the period end date, and at max 3 months from the report date.
    ## this is in line with SEC regulations for quarterly statements.
    min_shift_q_days, max_shift_q = 20, 3
    quarterly.loc[:,"date"] = np.nan
    quarterly["min_date"] = quarterly["datadate"].apply(
        lambda x: day_shift_nodt(x,min_shift_q_days))
    quarterly["max_date"] = quarterly["datadate"].apply(
        lambda x: month_shift_num(x,3,end=False))
    quarterly.loc[~quarterly["rdq"].isna(),"date"] = quarterly.loc[
        ~quarterly["rdq"].isna(),["rdq","min_date"]].max(axis=1)
    quarterly.loc[quarterly["rdq"].isna(),"date"] = quarterly.loc[
        quarterly["rdq"].isna(),"max_date"]
    quarterly.loc[:,"date"] = quarterly[["max_date","date"]].min(axis=1)
    del quarterly["min_date"], quarterly["max_date"]
    
    ## extract report dates
    report_dates = quarterly[["rdq","cusip","gvkey"]].dropna()
    report_dates.drop_duplicates(inplace=True, subset = ["gvkey","rdq"])
    report_dates["rdq"] = report_dates["rdq"].astype(int)
    report_dates["gvkey"] = report_dates["gvkey"].astype(int)
    report_dates.to_hdf(PATH_DATA+"report_dates.h5",key="data")
    del quarterly["cusip"]
    
    ## integer date representation:
    quarterly["date_int"] = quarterly["date"]
    quarterly["date_int"] = quarterly["date_int"].apply(lambda x: str(int(x)))
    quarterly["date_int"] = quarterly["date_int"].apply(lambda x: dt.datetime.strptime(x,"%Y%m%d"))
    date_0 = quarterly["date_int"].max()
    quarterly["date_int"] = (quarterly["date_int"]-date_0).dt.days
    del date_0
    
    '''
    Description of input variables:
    
    CUSIP     :	Merging Identifier between compustat and CRSP
    PERMNO    :	Unique Identifier CRSP
    acoq      :	Other current assets
    actq      :	Total current assets (cheq+rectq+invtq+acoq)
    altoq     :	Other long term assets (dcq (deffered charges)+aldo (long term assets 
                    discontinued operations)+ aodo (other assets including discontinued operations))
    aoq       :	Other assets (Various other items: 77)
    apq       :	Accounts payable 
    atq       :	Total assets
    capxy     :	Capital expenditure = net cash flow from investing activities EXCLUDING acquisitions
    ceqq      :	Common equity (req (retained earnings)+ capsq(Capital surplus)+
                               cstkq (Ordinary Stock capital))
    cheq      :	Cash and short-term investments (chq + ivstq)
    chq       :	Cash and cash equivalents
    cogsq     :	Cost of goods sold 
    conm      :	Company name
    conml     :	Company legal name
    consol    :	???
    costat    :	???
    cshiq     :	Common shares issued (cshoq + tstknq(Treasury stock))
    cshoq     :	Common shares outstanding
    cshprq    :	Common shares for calculation of earnings per share
    curcdq    :	Currency 
    cusip     :	Merging Identifier between compustat and CRSP
    datacqtr  :	calendar data year and quarter
    datadate  :	end of quarter date
    datafmt   :	???
    datafqtr  :	
    date      :	fiscal year and quarter
    dlcq      :	debt in current liabilities (dd1q(Long term debt due in one year)+\
                                             npq(notes payable, short term borrowings))
    dlttq     :	Total long term debt
    doq       :	Discontinued operations
    dpq       :	Depreciation and amortization
    dpretq    :	Depreciation and amortization of property
    drcq      :	Current deferred revenue (soon to be earned revenue)
    drltq     : ???
    epspiq    :	Earnings per share including extraordinaries
    esubq     : Equity in earnings, unconsolidated
    fqtr      :	Fiscal quarter
    fyearq    :	Fiscal year
    gdwliaq   :	Impairment (Verminderung) of goodwill after tax
    gdwlipq   :	Impairment (Verminderung) of goodwill before tax
    gdwlq     :	Goodwill (excess cost of equity of an aquired company)
    glivq     : Gains/Losses on investments
    gvkey     :	Unique identifier in Compustat
    ibmiiq    : Income from extraordinary items
    ibq       :	Income before extraordinaries
    indfmt    :	industry format
    intanq    :	Total intangible assets
    invtq     :	Inventories
    ipodate   :	Date of IPO
    lcoq      :	Other current liabilities (xaccq)
    lctq      :	Total current liabilities (apq + lcoq + dlcq + txpq)
    lltq      :	Total long term liabilities
    loq       :	Other liabilities
    ltq       :	Total liabilities (loq + lctq+ txditc (Deferred taxes and investment credit) + dlltq)
    niq       :	Net income
    nopiq     :	Non-operating income or expense
    oancfy    :	Net cash flow from operating activities annual (
                    apalch (Accounts Payable and Accrued Liabilities change)
                    recch (Accounts Receivable change)
                    aoloch (Assets and Liabilities - Other change) 
                    txdc (Deferred Taxes from cash flows) 
                    dpc (Depreciation and Amortization from cash flows)
                    esubc (Equity in Net Loss/Earnings) 
                    xidoc (Extraordinary Items & Discontinued Operations from cash flows) 
                    fopo (Funds from Operations - Other) 
                    IBC (Income Before Extraordinary Items from cash flows)
                    TXACH (Income Taxes - Accrued change) 
                    INVCH (Inventory change) 
                    SPPIV (Sale of PP&E and Investments gain)

    oibdpq    :	Operating income before depreciation
    piq       :	Pretax income
    popsrc    :	???
    ppegtq    :	Property, plant and equipment, gross
    ppentq    :	Property, plant and equipment, net (ppegtq + dpactq (accrued PPE))
    prchq     :	Quarterly price high
    prclq     :	Quarterly price low
    pstkq     :	Preferred stock capital
    pstkrq    :	Preferred stock redeemable
    rdq       :	Report date of quarterly earnings
    rectq     :	Accounts receivables total
    retq      :	Total RE property
    revtq     :	Total revenue (saleq + (other revenues))
    saleq     :	Total sales
    scstkcy   :	Sale of common stock, including conversions and issuance
    seqq      :	Total stockholders (shareholders) equity (ceq+pstkq)
    spiq      :	Unusual expenses or special items
    sretq     : Sale of property
    txdbq     : Deferred taxes from balance sheet
    txdiq     :	Deferred income taxes
    txpq      :	Income tax payable
    txtq      :	Total Income tax
    udmbq     :	Mortgage bonds
    uaptq     : Accounts payable
    ulcoq     :	Other current liabilities
    wcapq     :	Working capital
    xaccq     :	Accrued expenses
    xidoq     : extraordinary items
    xintq     :	Interest related expenses
    xoprq     :	operating expenses (cogsq + xsgaq)
    xrdq      :	Research and developement cost
    xsgaq     :	Selling, general and administrative cost
    '''
    quarterly.dropna(subset=["gvkey","date"],inplace=True)
    quarterly.set_index(["gvkey","date"],inplace=True)
    quarterly.sort_index(inplace=True)
    ## net operating cash flow quarterly
    # d(uaptq) + d(rectq) + d(aoq) + d(txbdq) + dpq + esupb + 
    # xidoc + ibmiiq + d(invtq) + d(txpg) +sretq +glivq
    quarterly.loc[:,"oancfq"] = \
        quarterly["niq"].replace(np.nan,0) + quarterly["dpq"].replace(np.nan,0)+ \
            delta_series(quarterly["txdbq"]) - delta_series(quarterly["rectq"])+\
            delta_series(quarterly["apq"]) - delta_series(quarterly["invtq"])-\
            delta_series(quarterly["ulcoq"]) + delta_series(quarterly["acoq"]) +\
            quarterly["xidoq"].replace(np.nan,0)
    
    
    '''
             NA    ZERO     mean        std      skew         kurt
    oancfq  0.0  0.1727  52.8877  3402.9245  275.8167  178787.4077
                      JB   99.5%       0.5%         min        max  median
    oancfq  2.941447e+10  3073.0 -1051.2353 -809853.355  2210029.0   0.263
    '''
    
    # alternative
    # quarterly.loc[:,"oancfq"] = quarterly["niq"].replace(np.nan,0) + quarterly["dpq"].replace(np.nan,0) - \
    #     (quarterly["rectq"]-quarterly["rectq"].shift(1)).replace(np.nan,0) +\
    #     (quarterly["apq"]-quarterly["apq"].shift(1)).replace(np.nan,0) 
    
    # ''' 
    #          NA      ZERO       mean         std        skew          kurt
    # oancfq  0.0  0.157994  50.907749  7801.69764  111.771529  83898.994596
    #                   JB     99.5%      0.5%        min        max  median
    # oancfq  8.150305e+09  3371.131 -1314.925 -2257443.0 -2257443.0   0.162
    # '''



    quarterly.loc[:,"acc"] = (quarterly.loc[:,"ibq"].replace(np.nan,0)-
                              quarterly.loc[:,"oancfq"])
    quarterly.loc[:,"acc"] = quarterly.loc[:,"acc"].replace([np.inf,-np.inf],np.nan)
    # series = windsorise(quarterly[["acc","date_int"]])
    # del quarterly["acc"]
    # quarterly = pd.merge(quarterly,series,left_index=True,right_index=True)
    ''' 
    #########################
    ########## acc ##########
    #########################

          NA   ZERO     mean        std      skew         kurt
    acc  0.0  0.223 -25.3445  3384.9156 -284.8905  186907.0699
                   JB     99.5%       0.5%        min         max  median
    acc  3.114827e+10  1068.503 -2013.0518 -2225204.0  810765.591  -0.064
    '''
    
    series = (quarterly["epspiq"]*quarterly["cshprq"]).groupby("gvkey").\
        apply(lambda x: x.rolling(4,1).mean()*4).replace([np.inf,-np.inf],np.nan).droplevel(0)
    series.sort_index(inplace=True)
    quarterly["oiq"] = series
    ''' 
    #########################
    ########## oiq ##########
    #########################

             NA   ZERO     mean       std     skew       kurt
    oiq  0.1661  0.006  34.2927  349.2586  21.3412  2106.5733
                   JB      99.5%      0.5%        min       max  median
    oiq  2.390126e+08  1524.9718 -249.8798 -26308.007  49383.96  0.7776
    '''

    quarterly.loc[:,"inc"] = quarterly["niq"].groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4).droplevel(0)
    ''' 
    #########################
    ########## inc ##########
    #########################

             NA    ZERO      mean        std     skew       kurt
    inc  0.1605  0.0004  140.8289  1411.1273  21.4901  2031.1079
                   JB   99.5%     0.5%         min        max  median
    inc  2.359713e+08  6211.0 -998.581 -105167.314  197612.09   3.262 
    '''
    
    quarterly.loc[:,"cf4qr"] = quarterly["oancfq"].groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4).\
        replace([np.inf,-np.inf],np.nan).droplevel(0)
    ''' 
    #########################
    ######### cf4qr #########
    #########################

            NA    ZERO      mean        std      skew       kurt
    cf4qr  0.0  0.1571  206.9778  6038.4426  187.9215  64477.427
                     JB   99.5%       0.5%         min        max  median
    cf4qr  1.252693e+10  9315.0 -1728.2443 -707895.048  2283642.0   2.191
    '''

    quarterly["cin"] = (delta_series(quarterly["ppegtq"])+quarterly["dpq"].replace(np.nan,0)/\
                          quarterly["saleq"].groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4).droplevel(0)).\
        replace([np.inf,-np.inf],np.nan)
        
    quarterly.loc[:,"cin"] = quarterly["cin"].clip(-10,10)
    ''' 
    #########################
    ########## cin ##########
    #########################

             NA    ZERO    mean     std    skew    kurt
    cin  0.1817  0.1497 -1.1648  3.7753 -0.7958  2.4132
                  JB  99.5%  0.5%   min   max  median
    cin  118502.5363   10.0 -10.0 -10.0  10.0     0.0
    '''

    
    quarterly["dsaleq"] = (rolling_series(quarterly["saleq"])/rolling_series(quarterly["saleq"]).shift(1)-1).\
        replace([np.inf,-np.inf],np.nan)
        
    quarterly["rxsgaq"] = rolling_series(quarterly["xsgaq"])
    
    quarterly["noa"]  = (quarterly["rectq"].replace(np.nan,0)+quarterly["invtq"].replace(np.nan,0)+\
                           quarterly["aoq"].replace(np.nan,0)+quarterly["acoq"].replace(np.nan,0)+\
                           quarterly["ppentq"].replace(np.nan,0)+quarterly["intanq"].replace(np.nan,0)-\
                           quarterly["apq"].replace(np.nan,0)-quarterly["lcoq"].replace(np.nan,0)-\
                           quarterly["loq"].replace(np.nan,0))
    ''' 
    noa: 

    NA 0.0 % 
    ZERO 23.651 %; 
    mean 2045.90995 
    min: -1035614.0 
    max: 4014007.0 
    std 24392.6547 
    1% -57.9218 
    99% 34660.0787 
    '''
    quarterly["capxq"] = (quarterly["ppentq"]+ rolling_series(quarterly["dpq"]))-\
                          quarterly["ppentq"].groupby("gvkey").shift(4).\
          replace([np.inf,-np.inf],np.nan) 
    quarterly["rtxtq"] = rolling_series(quarterly["txtq"])
    
    quarterly["redeq"] = rolling_series(quarterly["xrdq"])+rolling_series(quarterly["xsgaq"])
    '''
    #########################
    ######### redeq #########
    #########################

            NA    ZERO      mean        std     skew       kurt
    redeq  0.0  0.3013  330.7003  2367.1952  28.0549  1861.1435
                     JB       99.5%  0.5%    min       max  median
    redeq  3.048381e+08  12645.9294   0.0 -283.0  318049.0  11.108
    '''
    # quarterly["ACA"] = (quarterly["xaccq"].abs()/quarterly["atq"]).replace([np.inf,-np.inf],np.nan)
    # ''' 
    # ACA: 

    # NA 91.556 % 
    # ZERO 0.659 %; 
    # mean 1.27299 
    # min: -0.024 
    # max: 9915.0 
    # std 48.2506 
    # '''
    
    
    columns_quarterly_1 = [
        'ACA', 'ACV', 'ACW', 'AG1Y', 'C', 'CEI', 'CFD', 'CFV', 'CIN', 'CP', 'CR', 'DCE', 
        'DCR', 'DCSE', 'DD', 'DE', 'DGMDS', 'DI', 'DLTD', 'DNOA', 'DPPE', 'DQR', 'DRD', 
        'DS', 'DSDAR', 'DSDI', 'DSDSGA', 'DSI', 'DTAX', 'EIS', 'EV', 'FSS', 
        'GP', 'IDAT', 'IDPM', 'L', 'OC', 'OP', 'QR', 'RDS', 
        'RE', 'ROA', 'ROE', 'ROI', 'RS', 'SC', 'SD', 'SI', 'SR', 'TANG', 'TIBI'
        ]
        
    if not os.path.exists(PATH_DATA+"quarterly_P3_s1.h5") or "FULL" in mode:
        quarterly["ACA"] = windsorise((quarterly["acc"].apply(abs)/quarterly["atq"]).replace([np.inf,-np.inf],np.nan))
        ''' 
        #########################
        ########## ACA ##########
        #########################
    
                 NA    ZERO   mean     std   skew     kurt
        ACA  0.2634  0.0117  0.046  0.0832  6.285  71.7844
                       JB   99.5%  0.5%  min     max  median
        ACA  1.380919e+07  0.5982   0.0  0.0  5.3938  0.0226
        '''
    
        quarterly["ACW"] = windsorise((quarterly["acc"]/quarterly["wcapq"]).\
            replace([-np.inf,np.inf],np.nan))
        ''' 
        #########################
        ########## ACW ##########
        #########################
    
                 NA    ZERO    mean     std    skew     kurt
        ACW  0.4017  0.0072 -0.0553  1.6923 -0.5313  48.9819
                       JB   99.5%    0.5%      min     max  median
        ACW  2.868566e+06  9.5264 -9.7215 -35.2152  32.583 -0.0315
        '''
    
        quarterly["AG1Y"] = delta_series_relative(quarterly["atq"],shift=4).clip(-10,10)
        ''' 
        #########################
        ########## AG1Y #########
        #########################
    
                  NA    ZERO    mean     std     skew      kurt
        AG1Y  0.3278  0.0002  0.0216  0.5627 -10.2146  156.0973
                        JB   99.5%    0.5%   min   max  median
        AG1Y  3.473414e+07  0.8921 -2.7581 -10.0  10.0  0.0606
        '''
    
        quarterly["BM"] = quarterly["ceqq"]+0
        ''' 
        --- FLAG: Basis for monthly predictor---
        #########################
        ########### BM ##########
        #########################
         
                NA    ZERO       mean        std     skew      kurt
        BM  0.2481  0.0002  1187.4101  7366.7334  18.9267  617.9695
                      JB      99.5%   0.5%       min       max  median
        BM  1.246923e+08  38941.883 -529.0 -139965.0  539883.0  73.254
        '''
    
        quarterly["C"] = (quarterly["cheq"]/quarterly["atq"]).replace([np.inf,-np.inf],np.nan).clip(-10,10)
        ''' 
        #########################
        ########### C ###########
        #########################
    
               NA   ZERO    mean     std   skew    kurt
        C  0.2716  0.007  0.1734  0.2242  1.904  6.1972
                     JB   99.5%  0.5%    min   max  median
        C  1.077604e+06  0.9721   0.0 -0.904  10.0  0.0761
        '''
    
        quarterly["CP"] =  windsorise((quarterly["oiq"]/quarterly["cheq"]).replace([np.inf,-np.inf],np.nan))
        ''' 
        #########################
        ########### CP ##########
        #########################
    
                NA    ZERO   mean      std     skew     kurt
        CP  0.2836  0.0042 -1.453  30.6454 -10.0703  180.331
                      JB    99.5%   0.5%       min       max  median
        CP  3.549683e+07  86.9916 -182.7 -880.7094  327.8623  0.2299
        '''
    
        quarterly["IDAT"] = (quarterly["saleq"].groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4).droplevel(0)/\
                             quarterly["atq"]).replace([np.inf,-np.inf],np.nan)
        quarterly["IDAT"] = (quarterly["IDAT"]/quarterly["IDAT"].shift(1)-1).replace([np.inf,-np.inf],np.nan)
        quarterly["IDAT"] = windsorise(quarterly["IDAT"])
        ''' 
        #########################
        ########## IDAT #########
        #########################
    
                  NA  ZERO   mean     std    skew     kurt
        IDAT  0.3212   0.0  0.019  0.2338  4.2326  44.2685
                        JB   99.5%    0.5%  min     max  median
        IDAT  6.876118e+06  1.4221 -0.7213 -1.0  6.3589  0.0053
        '''     
                
        quarterly["DI"] = (delta_series(quarterly["invtq"]).replace([np.inf,-np.inf],np.nan)/\
            quarterly["atq"]).replace([np.inf,-np.inf],np.nan).clip(-1,1)
        ''' 
        #########################
        ########### DI ##########
        #########################
    
                NA    ZERO   mean     std    skew      kurt
        DI  0.2634  0.2489  0.001  0.0402 -5.4202  152.1568
                      JB   99.5%    0.5%  min  max  median
        DI  1.623748e+07  0.1488 -0.1674 -1.0  1.0     0.0
        '''
    
        quarterly["GP"] = 1-(rolling_series(quarterly["cogsq"],replace=None)/rolling_series(quarterly["saleq"])).\
            replace([np.inf,-np.inf],np.nan).clip(-10,10)
        ''' 
        #########################
        ########### GP ##########
        #########################
    
                NA    ZERO    mean     std    skew    kurt
        GP  0.2065  0.0001  0.0552  1.4389 -5.4061  33.303
                      JB  99.5%  0.5%   min   max  median
        GP  8.963428e+06  0.992 -10.0 -10.0  10.0  0.3252
        '''
    
        quarterly["IDPM"] = delta_series(quarterly["GP"]).replace([np.inf,-np.inf],np.nan)
        ''' 
        #########################
        ########## IDPM #########
        #########################
        
               NA    ZERO    mean     std    skew      kurt
        IDPM  0.0  0.2437  0.0061  0.4294  1.9881  484.4492
                        JB   99.5%    0.5%   min   max  median
        IDPM  3.027785e+07  1.2738 -1.0926 -20.0  20.0     0.0
        '''
    
        quarterly["DCSE"] = windsorise((quarterly["ceqq"]/quarterly["ceqq"].groupby("gvkey").\
            shift(1) -1).replace([np.inf,-np.inf],np.nan))
        ''' 
        #########################
        ########## DCSE #########
        #########################
        
                  NA    ZERO    mean     std    skew     kurt
        DCSE  0.2867  0.0007  0.0289  0.6924  2.7146  90.2537
                        JB   99.5%    0.5%      min      max  median
        DCSE  7.107556e+06  3.9502 -3.2653 -14.7216  23.7709  0.0154
        '''  
    
        quarterly["EP"] = quarterly["oiq"]+0
        ''' 
        --- FLAG: Basis for monthly predictor---
        #########################
        ########## oiq ##########
        #########################
        
                 NA   ZERO      mean        std     skew       kurt
        oiq  0.1661  0.006  137.1709  1397.0345  21.3412  2106.5733
                       JB      99.5%      0.5%         min        max  median
        oiq  2.390126e+08  6099.8873 -999.5191 -105232.028  197535.84  3.1105
        '''
    
        quarterly["ei"] = False
        quarterly.loc[quarterly["oiq"]>=quarterly["oiq"].groupby("gvkey").shift(1),"ei"] = True
        quarterly["eis_start"] = quarterly["ei"].ne(quarterly["ei"].groupby("gvkey").shift(1))
        quarterly["eis_id"] = quarterly.groupby("gvkey")["eis_start"].cumsum()
        quarterly["EIS"] = (quarterly.groupby(["eis_id","gvkey"]).cumcount()+1)*quarterly["ei"]
        del quarterly["ei"], quarterly["eis_start"], quarterly["eis_id"]
        quarterly["EIS"] = np.log(quarterly["EIS"]+1)
        
        ''' 
        #########################
        ########## EIS ##########
        #########################
    
              NA    ZERO    mean     std    skew     kurt
        EIS  0.0  0.5279  2.0049  4.1878  6.3898  87.1405
                       JB  99.5%  0.5%  min    max  median
        EIS  1.506752e+07   25.0   0.0  0.0  144.0     0.0
        '''
        
        quarterly["OP"] = 1-(rolling_series(quarterly["xoprq"].replace(np.nan,0)+quarterly["dpq"].replace(np.nan,0))/\
            rolling_series(quarterly["saleq"])).replace([np.inf,-np.inf],np.nan).clip(-10,1)
        ''' 
        #########################
        ########### OP ##########
        #########################
    
                NA    ZERO    mean     std    skew     kurt
        OP  0.1817  0.0001 -0.1856  1.5768 -5.1245  27.3502
                     JB  99.5%  0.5%   min  max  median
        OP  7878699.906    1.0 -10.0 -10.0  1.0  0.0892
        '''
    
        quarterly["OC"] = (quarterly["xsgaq"].groupby("gvkey").apply(lambda x: x.rolling(12,4).mean()*12).\
                           droplevel(0)/quarterly["atq"]).replace([np.inf,-np.inf],np.nan)
        quarterly["OC"] = windsorise(quarterly["OC"])
        ''' 
        #########################
        ########### OC ##########
        #########################
    
                NA  ZERO    mean     std    skew     kurt
        OC  0.4209   0.0  0.9479  1.3644  5.6798  54.2023
                      JB   99.5%    0.5%     min      max  median
        OC  1.097485e+07  9.2335  0.0151  0.0074  34.0205  0.6027
        '''   
        
        quarterly["DGMDS"] = (quarterly["GP"]/quarterly["GP"].groupby("gvkey").shift(1)-\
                              quarterly["dsaleq"])
        quarterly["DGMDS"] = windsorise(quarterly["DGMDS"])
        ''' 
        #########################
        ######### DGMDS #########
        #########################
    
                   NA  ZERO    mean     std    skew     kurt
        DGMDS  0.2226   0.0  0.9343  0.7283 -3.0558  64.6246
                         JB   99.5%    0.5%      min      max  median
        DGMDS  6.026624e+06  4.0944 -3.5679 -17.0833  20.3534  0.9775
        '''
    
        quarterly["DSDI"] = (quarterly["dsaleq"]-quarterly["DI"]).\
            replace([np.inf,-np.inf],np.nan)
        quarterly["DSDI"] = windsorise(quarterly["DSDI"])
        ''' 
        #########################
        ########## DSDI #########
        #########################
    
                NA    ZERO    mean     std     skew      kurt
        DSDI  0.29  0.0023  0.0364  0.2259  14.2651  730.4457
                        JB   99.5%    0.5%  min      max  median
        DSDI  9.385616e+07  1.3114 -0.5825 -1.0  18.1336  0.0188
        '''
    
        quarterly["DSDAR"] = (quarterly["dsaleq"] -\
                quarterly["rectq"]/\
                quarterly["rectq"].groupby("gvkey").shift(1)).replace([np.inf,-np.inf],np.nan)
        quarterly["DSDAR"] = windsorise(quarterly["DSDAR"])
        ''' 
        #########################
        ######### DSDAR #########
        #########################
    
                   NA    ZERO    mean     std    skew     kurt
        DSDAR  0.3636  0.0002 -1.0582  0.5469 -6.6901  73.5212
                         JB   99.5%    0.5%     min     max  median
        DSDAR  1.519490e+07  0.0304 -4.8796 -14.639  1.5161 -0.9991
        '''    
                
        quarterly["DSDSGA"] = (quarterly["dsaleq"] - \
                quarterly["rxsgaq"]/quarterly["rxsgaq"].groupby("gvkey").shift(1)).\
            replace([np.inf,-np.inf],np.nan)
        quarterly["DSDSGA"] = windsorise(quarterly["DSDSGA"])
        ''' 
        #########################
        ######### DSDSGA ########
        #########################
    
                   NA    ZERO   mean     std    skew     kurt
        DSDSGA  0.364  0.0001 -0.989  0.1617  2.5733  30.6249
                          JB   99.5%    0.5%     min     max  median
        DSDSGA  3.294820e+06  0.0128 -1.5732 -5.8682  1.5911 -0.9991
        '''
    
        quarterly["SI"]  = (rolling_series(quarterly["saleq"])/rolling_series(quarterly["invtq"])).\
            replace([np.inf,-np.inf],np.nan).clip(0,10)
        ''' 
        #########################
        ########### SI ##########
        #########################
        
                NA    ZERO    mean     std    skew    kurt
        SI  0.4236  0.0026  3.6808  3.3698  1.0054 -0.5485
                    JB  99.5%    0.5%  min   max  median
        SI  30140.4914   10.0  0.0084  0.0  10.0  2.1026
        '''
    
        quarterly["DSI"] = (quarterly["SI"].replace(np.nan,0)/quarterly["SI"].groupby("gvkey").shift(1)-1).\
            replace([np.inf,-np.inf],np.nan).clip(-1,10)
            
        ''' 
        #########################
        ########## DSI ##########
        #########################
    
                 NA    ZERO    mean     std     skew      kurt
        DSI  0.4366  0.0907  0.0127  0.3168  17.7635  488.8084
                       JB   99.5%    0.5%  min   max  median
        DSI  1.064333e+08  1.0845 -0.9679 -1.0  10.0     0.0
        '''
    
        
        quarterly["DRD"] = delta_series_relative(quarterly["redeq"],replace=np.nan).clip(-1,10)
        ''' 
        #########################
        ########## DRD ##########
        #########################
    
                 NA    ZERO    mean    std    skew     kurt
        DRD  0.3073  0.0382  0.0257  0.189  2.3174  75.6587
                       JB  99.5%    0.5%  min   max  median
        DRD  5.732121e+06    1.0 -0.7779 -1.0  10.0  0.0167
        '''
    
        quarterly["RDMCAP"] =  quarterly["redeq"]
        ''' 
        --- FLAG: Basis for monthly predictor---
        #########################
        ######### RDMCAP ########
        #########################
        
                 NA    ZERO      mean        std     skew       kurt
        RDMCAP  0.0  0.3013  330.7003  2367.1952  28.0549  1861.1435
                          JB       99.5%  0.5%    min       max  median
        RDMCAP  3.048381e+08  12645.9294   0.0 -283.0  318049.0  11.108
        '''
    
        quarterly["RDS"] =  (quarterly["redeq"]/rolling_series(quarterly["saleq"])
                               ).replace([np.inf,-np.inf],np.nan).clip(-10,10)
        
        ''' 
        #########################
        ########## RDS ##########
        #########################
    
                 NA    ZERO          mean           std      skew       kurt
        RDS  0.1817  0.1414  2.452472e+10  2.681878e+13  1093.541  1195832.0
                       JB    99.5%  0.5%        min           max  median
        RDS  3.640646e+11  52.1018   0.0 -6811.2222  2.932744e+16  0.2271
        '''
    
        quarterly["ROA"] = (quarterly["oiq"] / quarterly["atq"].abs()).replace([np.inf,-np.inf],np.nan).clip(-10,10)
        ''' 
        #########################
        ########## ROA ##########
        #########################
    
                 NA    ZERO    mean     std    skew      kurt
        ROA  0.2693  0.0044 -0.0916  0.7137 -6.7808  113.9463
                       JB   99.5%  0.5%   min   max  median
        ROA  1.795395e+07  0.5991 -4.27 -10.0  10.0  0.0187
        '''
    
        quarterly["EV"] = ((quarterly["epspiq"]*quarterly["cshprq"]).groupby("gvkey").\
            apply(lambda x: x.rolling(16,4).mean()) /\
                (quarterly["epspiq"]*quarterly["cshprq"]).groupby("gvkey").\
            apply(lambda x: x.rolling(16,4).std())).replace([np.inf,-np.inf],np.nan).droplevel(0).\
            clip(-10,10)
        ''' 
        #########################
        ########### EV ##########
        #########################
    
                NA    ZERO    mean     std    skew    kurt
        EV  0.2185  0.0001  1.0092  2.0752  1.0617  3.4776
                     JB  99.5%    0.5%   min   max  median
        EV  303620.7521   10.0 -4.2025 -10.0  10.0  0.6229
        '''
    
        quarterly["ROE"] = windsorise((quarterly["oiq"]/quarterly["ceqq"]).replace([np.inf,-np.inf],np.nan))
        ''' 
        #########################
        ########## ROE ##########
        #########################
    
                NA    ZERO    mean     std    skew     kurt
        ROE  0.254  0.0045 -0.0102  1.6256 -0.5353  57.6174
                       JB   99.5%    0.5%      min      max  median
        ROE  3.395387e+06  8.5283 -9.0899 -29.6391  27.9998  0.0855
        '''
            
        quarterly["RS"] = ((quarterly["saleq"]-quarterly["saleq"].groupby("gvkey").\
            apply(lambda x: x.rolling(16,4).mean()).droplevel(0)) / quarterly["saleq"].groupby("gvkey").\
            apply(lambda x: x.rolling(16,4).std()).droplevel(0)).replace([np.inf,-np.inf],np.nan).\
            clip(-10,10)
        ''' 
        #########################
        ########### RS ##########
        #########################
    
                NA  ZERO    mean     std    skew    kurt
        RS  0.2366   0.0  0.6285  1.1642 -0.3024 -0.5167
                    JB   99.5%    0.5%   min   max  median
        RS -191857.772  3.1876 -2.3431 -3.75  3.75  0.7898
        '''
    
        quarterly["SC"] = (rolling_series(quarterly["saleq"])/\
            quarterly["cheq"]).replace([np.inf,-np.inf],np.nan).clip(0,1000)
        ''' 
        #########################
        ########### SC ##########
        #########################
    
                NA    ZERO     mean       std    skew     kurt
        SC  0.2784  0.0213  53.3106  149.5166  4.8954  25.6455
                      JB   99.5%  0.5%  min     max  median
        SC  7.215682e+06  1000.0   0.0  0.0  1000.0  7.6313
        '''
    
        quarterly["SR"] = (rolling_series(quarterly["saleq"])/\
            quarterly["rectq"]).replace([np.inf,-np.inf],np.nan).clip(0,100)
        ''' 
        #########################
        ########### SR ##########
        #########################
    
                NA    ZERO     mean      std    skew     kurt
        SR  0.3145  0.0102  10.6494  18.0417  3.7383  14.3068
                    JB  99.5%  0.5%  min    max  median
        SR  4092061.92  100.0   0.0  0.0  100.0  5.8112
        '''
    
        quarterly["DS"] = quarterly["dsaleq"].clip(-1,10)
        ''' 
        #########################
        ########### DS ##########
        #########################
    
                NA    ZERO    mean     std     skew      kurt
        DS  0.1817  0.0142  0.0485  0.5522  13.7283  230.5936
                      JB   99.5%  0.5%  min   max  median
        DS  5.976013e+07  2.2848  -1.0 -1.0  10.0  0.0188
        '''
    
        quarterly["SP"] = rolling_series(quarterly["saleq"],replace=None).replace([np.inf,-np.inf],np.nan)
        ''' 
        --- FLAG: Basis for monthly predictor---
        #########################
        ########### SP ##########
        #########################
    
                NA    ZERO       mean         std     skew      kurt
        SP  0.1584  0.0233  2263.9887  12918.7548  17.3133  465.1808
                      JB      99.5%  0.5%        min       max   median
        SP  1.011480e+08  75232.125   0.0 -50077.988  783220.0  121.418
        '''
          
        quarterly["ACV"] = windsorise((quarterly["acc"].groupby("gvkey").\
                                       apply(lambda x: x.rolling(16,4).std()).droplevel(0)/
                            quarterly["atq"]).replace([np.inf,-np.inf],np.nan))
        ''' 
        #########################
        ########## ACV ##########
        #########################
    
                 NA    ZERO    mean     std     skew       kurt
        ACV  0.2708  0.0042  0.0619  0.1865  23.4736  1042.0267
                       JB  99.5%  0.5%  min      max  median
        ACV  1.974672e+08  1.022   0.0  0.0  13.4589  0.0288
        '''
    
        quarterly["CFV"] = (quarterly["oancfq"].groupby("gvkey").apply(lambda x: x.rolling(16,4).mean())/\
                            quarterly["oancfq"].groupby("gvkey").apply(lambda x: x.rolling(16,4).std())).\
                                  replace([np.inf,-np.inf],np.nan).droplevel(0)
        quarterly["CFV"] = windsorise(quarterly["CFV"])
        ''' 
        #########################
        ########## CFV ##########
        #########################
    
                 NA  ZERO    mean     std    skew     kurt
        CFV  0.1859   0.0  0.4156  1.1813  2.6398  25.6693
                       JB   99.5%    0.5%    min      max  median
        CFV  3.077513e+06  5.6004 -2.9125 -5.815  35.6055  0.2819
        '''
    
        quarterly["CFD"] = (quarterly["cf4qr"]/ (quarterly["dlttq"]+ quarterly["dlcq"])).\
            replace([np.inf,-np.inf],np.nan)
        quarterly["CFD"] = windsorise(quarterly["CFD"])
        ''' 
        #########################
        ########## CFD ##########
        #########################
    
                 NA    ZERO    mean     std   skew      kurt
        CFD  0.3899  0.0004 -0.4111  32.701 -3.375  243.8458
                       JB     99.5%      0.5%       min        max  median
        CFD  1.743914e+07  113.4459 -150.0486 -1305.041  1046.0817  0.1531
        '''
    
        quarterly["CFP"] = quarterly["cf4qr"]+0
        ''' 
        --- FLAG: Basis for monthly predictor---
        #########################
        ########## CFP ##########
        #########################
    
              NA    ZERO      mean       std      skew        kurt
        CFP  0.0  0.1575  192.7642  6159.712 -192.1239  68430.7576
                       JB      99.5%       0.5%        min        max  median
        CFP  1.315652e+10  9340.8941 -1500.0529 -2326918.0  707937.86   1.201
        '''
    
        quarterly["DTAX"]= ((quarterly["rtxtq"]-quarterly["rtxtq"].groupby("gvkey").shift(1))/\
                            quarterly["rtxtq"].groupby("gvkey").shift(1).abs()-1).replace([np.inf,-np.inf],np.nan)
        quarterly["DTAX"] = quarterly["DTAX"].clip(-10,10)
        ''' 
        #########################
        ########## DTAX #########
        #########################
    
                  NA    ZERO   mean     std    skew     kurt
        DTAX  0.3105  0.0033 -0.921  1.6774  1.1821  23.4301
                        JB  99.5%  0.5%   min   max  median
        DTAX  1.584327e+06   10.0 -10.0 -10.0  10.0 -0.9832
        '''
    
        quarterly["CIN"] = (quarterly["cin"]-quarterly["cin"].groupby("gvkey").shift(1).groupby("gvkey").\
            apply(lambda x: x.rolling(3,1).mean()).droplevel(0)).replace([np.inf,-np.inf],np.nan)
        ''' 
        #########################
        ########## CIN ##########
        #########################
    
                 NA    ZERO    mean    std    skew     kurt
        CIN  0.1975  0.1224  0.0263  3.281 -1.3373  14.4056
                       JB    99.5%     0.5%   min   max  median
        CIN  1.130020e+06  13.3333 -19.7002 -20.0  20.0     0.0
        '''
            
        quarterly["CR"] = (quarterly["actq"] / quarterly["lctq"]).replace([np.inf,-np.inf],np.nan)
        quarterly["CR"] = windsorise(quarterly["CR"])
        
        ''' 
        #########################
        ########### CR ##########
        #########################
    
                NA  ZERO    mean     std    skew     kurt
        CR  0.4017   0.0  3.1822  4.7827  5.9658  49.0024
                      JB    99.5%    0.5%     min      max  median
        CR  1.146936e+07  36.1861  0.0571  0.0036  87.5794  1.9466
        '''
    
    
        quarterly["DCR"] =(quarterly["CR"] / quarterly["CR"].shift(1)-1).replace([np.inf,-np.inf],np.nan)
        ''' 
        #########################
        ########## DCR ##########
        #########################
    
                 NA    ZERO    mean      std      skew        kurt
        DCR  0.4358  0.0001  0.2702  11.6794  166.3439  42161.3332
                       JB   99.5%    0.5%     min        max  median
        DCR  9.306238e+09  5.7875 -0.7738 -0.9994  4452.7395 -0.0075
        '''
    
        quarterly["DPPE"] = (rolling_series(quarterly["dpq"])/quarterly["ppentq"]).replace([np.inf,-np.inf],np.nan)
        quarterly["DPPE"] = windsorise(quarterly["DPPE"])
        ''' 
        #########################
        ########## DPPE #########
        #########################
    
                  NA    ZERO    mean     std    skew      kurt
        DPPE  0.3055  0.0748  0.3396  0.8316  10.757  178.3655
                        JB   99.5%  0.5%  min      max  median
        DPPE  3.886063e+07  5.5714   0.0  0.0  26.8295   0.159
        '''
        
        # quarterly["DCE"] = (quarterly["capxy"] / quarterly["capxy"].shift(4)-1).replace([np.inf,-np.inf],np.nan).clip(-10,10)
        
        quarterly["DCE"] = (quarterly["capxq"]/quarterly["capxq"].groupby("gvkey").shift(1)-1).clip(-10,10)
        ''' 
        #########################
        ########## DCE ##########
        #########################
    
                 NA   ZERO    mean     std    skew     kurt
        DCE  0.4036  0.002  0.0387  1.6295  0.6256  24.0452
                       JB  99.5%  0.5%   min   max  median
        DCE  1.376728e+06   10.0 -10.0 -10.0  10.0  0.0019
        '''
    
        # quarterly["CEI"] = ((quarterly["ppegtq"].groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4)+\
        #                        quarterly["invtq"]).groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4) /\
        #                       (quarterly["ppegtq"].groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4)+\
        #                        quarterly["invtq"]).groupby("gvkey").apply(lambda x: x.rolling(4,1).mean()*4).shift(1)-1).\
        #     replace([np.inf,-np.inf],np.nan)
        
        quarterly["CEI"] = ((quarterly["capxq"].replace(np.nan,0)+delta_series(quarterly["invtq"],4))/\
            quarterly["atq"]).replace([np.inf,-np.inf],np.nan)
        quarterly["CEI"] = windsorise(quarterly["CEI"])
            
        ''' 
        #########################
        ########## CEI ##########
        #########################
    
                 NA    ZERO    mean     std    skew     kurt
        CEI  0.2634  0.0987  0.0514  0.1416 -2.9922  43.3013
                       JB   99.5%    0.5%     min     max  median
        CEI  4.634557e+06  0.5383 -0.6219 -3.3276  0.7455  0.0337
        '''
    
        quarterly["L"]   =  (quarterly["ceqq"]/quarterly["atq"]).replace([np.inf,-np.inf],np.nan).clip(-10,10)
        ''' 
        #########################
        ########### L ###########
        #########################
    
              NA    ZERO    mean     std    skew      kurt
        L  0.269  0.0001  0.3515  0.7467 -9.2985  113.7507
                     JB   99.5%    0.5%   min     max  median
        L  2.780174e+07  0.9853 -4.3388 -10.0  9.4124  0.4245
        '''
    
        quarterly["DLTD"]= (quarterly["ltq"] / quarterly["ltq"].groupby("gvkey").shift(4)-1).\
            replace([np.inf,-np.inf],np.nan).clip(-10,10)
        ''' 
        #########################
        ########## DLTD #########
        #########################
    
                 NA    ZERO    mean     std    skew     kurt
        DLTD  0.332  0.0003  0.3054  1.1376  5.9917  42.9558
                        JB  99.5%   0.5%   min   max  median
        DLTD  1.117660e+07   10.0 -0.824 -10.0  10.0  0.0666
        '''
    
        # quarterly["dp_ppe"] = (quarterly["dpq"]/quarterly["ppentq"]).replace([np.inf,-np.inf],np.nan)
        quarterly["DD"]  = (quarterly["dpq"]/quarterly["dpq"].groupby("gvkey").shift(4)-1).replace([np.inf,-np.inf],np.nan).\
            clip(-10,10)
        ''' 
        #########################
        ########### DD ##########
        #########################
    
                NA    ZERO    mean     std    skew     kurt
        DD  0.4316  0.0077  0.3123  1.1819  5.6069  39.7014
                      JB  99.5%    0.5%   min   max  median
        DD  9.891412e+06   10.0 -0.9594 -10.0  10.0  0.0728
        '''
    
        quarterly["QR"]  = ((quarterly["cheq"] + quarterly["rectq"])/quarterly["lctq"]).\
            replace([np.inf,-np.inf],np.nan).clip(-1,10)
        ''' 
        #########################
        ########### QR ##########
        #########################
    
                NA    ZERO    mean     std    skew    kurt
        QR  0.4116  0.0007  2.0193  2.2966  2.2504  4.6019
                      JB  99.5%    0.5%  min   max  median
        QR  1.330992e+06   10.0  0.0153 -1.0  10.0  1.1909
        '''
    
        quarterly["DQR"] = (quarterly["QR"]/quarterly["QR"].groupby("gvkey").shift(4)-1).\
            replace([np.inf,-np.inf],np.nan).clip(-10,10)
        ''' 
        #########################
        ########## DQR ##########
        #########################
    
                 NA   ZERO    mean     std   skew   kurt
        DQR  0.4689  0.011  0.2106  1.2383  5.527  36.87
                       JB  99.5%    0.5%   min   max  median
        DQR  9.502463e+06   10.0 -0.9424 -10.0  10.0  -0.011
        '''
    
        quarterly["ROI"] = (rolling_series(quarterly["oibdpq"]) /\
                              (quarterly["ceqq"]+quarterly["ltq"]-quarterly["cheq"])).\
            replace([np.inf,-np.inf],np.nan)
        quarterly["ROI"] = windsorise(quarterly["ROI"])
        ''' 
        #########################
        ########## ROI ##########
        #########################
    
                 NA    ZERO    mean    std     skew      kurt
        ROI  0.2769  0.0736 -0.1747  1.825 -10.4249  143.8093
                       JB   99.5%     0.5%      min     max  median
        ROI  3.504302e+07  1.6096 -12.4115 -51.5667  7.0608  0.0817
        '''
    
        quarterly["TANG"] = ((quarterly["cheq"].replace(np.nan,0)+.715*quarterly["rectq"].replace(np.nan,0)+\
                                .547*quarterly["invtq"].replace(np.nan,0)+.535*quarterly["ppentq"].replace(np.nan,0))/\
                               quarterly["atq"].replace(np.nan,0)).replace([np.inf,-np.inf],np.nan)
        quarterly["TANG"] = windsorise(quarterly["TANG"])
        ''' 
        #########################
        ########## TANG #########
        #########################
    
                  NA    ZERO    mean     std    skew    kurt
        TANG  0.2634  0.0003  0.5098  0.1919 -0.3507  0.5534
                       JB   99.5%    0.5%  min     max  median
        TANG -119020.7611  0.9738  0.0062  0.0  1.4515   0.531
        '''
    
        quarterly["TIBI"] = ((1+rolling_series(quarterly["txtq"])/quarterly["oiq"])/1).\
            replace([np.inf,-np.inf],np.nan)
        quarterly["TIBI"] = windsorise(quarterly["TIBI"])
        ''' 
        #########################
        ########## TIBI #########
        #########################
    
                  NA  ZERO    mean     std    skew     kurt
        TIBI  0.1721   0.0  1.3543  0.7947  1.0735  40.9038
                        JB   99.5%    0.5%      min      max  median
        TIBI  2.588589e+06  5.4459 -2.4602 -10.3605  24.1595  1.3258
        '''
            
        quarterly["DNOA"] = (quarterly["noa"]/quarterly["noa"].groupby("gvkey").shift(4)-1).\
            replace([np.inf,-np.inf],0).clip(-10,10)
        ''' 
        #########################
        ########## DNOA #########
        #########################
    
                 NA    ZERO    mean     std    skew     kurt
        DNOA  0.278  0.0439  0.1766  1.5606  0.9642  26.1104
                        JB  99.5%  0.5%   min   max  median
        DNOA  1.633579e+06   10.0 -10.0 -10.0  10.0  0.0378
        '''
            
        quarterly["FSS"]  = 0
        quarterly.loc[(quarterly["inc"]>0) | (quarterly["inc"]>quarterly["inc"].groupby("gvkey").\
                                                  shift(4)),"FSS"] +=1
        quarterly.loc[(quarterly["cf4qr"]>0),"FSS"] +=1
        quarterly.loc[(quarterly["ROA"]>quarterly["ROA"].groupby("gvkey").shift(4)),"FSS"] +=1
        quarterly.loc[(quarterly["cf4qr"]> quarterly["inc"]),"FSS"] +=1
        quarterly.loc[(quarterly["L"]>quarterly["L"].groupby("gvkey").shift(4)),"FSS"] +=1
        quarterly.loc[(quarterly["CR"]>quarterly["CR"].groupby("gvkey").shift(4)),"FSS"] +=1
        quarterly.loc[rolling_series(quarterly["cshiq"].replace(np.nan,0))==0,"FSS"] +=1
        quarterly.loc[quarterly["GP"]>quarterly["GP"].groupby("gvkey").shift(4),"FSS"] +=1
        quarterly.loc[quarterly["IDAT"]>quarterly["IDAT"].groupby("gvkey").shift(4),"FSS"] +=1
        '''
        #########################
        ########## FSS ##########
        #########################
    
              NA   ZERO   mean     std  skew    kurt
        FSS  0.0  0.007  3.926  2.1435  0.13 -0.9341
                     JB  99.5%  0.5%  min  max  median
        FSS -235428.616    9.0   0.0  0.0  9.0     4.0
        '''
        
        ########################################
        ############# Annual based #############
        ########################################
        
        annually = pd.read_csv(
            PATH_DATA+"annually.txt", delimiter = "\t")
        
        ## dropping duplicate observations based on gvkey, datadate and NA_row.
        shape_before = annually.shape[0]
        annually["NA_row"] = annually.isna().sum(axis=1)
        annually.sort_values(by=["gvkey","datadate","NA_row","fyear"],ascending=True,inplace=True)
        annually.drop_duplicates(subset = ["gvkey","datadate"],keep="first",inplace=True)
        del annually["NA_row"]
        print(f"{shape_before-annually.shape[0]:8d} rows dropped, duplicate gvkey and datadate.",
              f"{annually.shape[0]:d} rows left.")
        '''
        43987 rows dropped, duplicate gvkey and datadate. 493987 rows left.
        '''
        
        
        annually.set_index(["gvkey","datadate"],inplace=True)
        
        
        
        annually["DE"] = delta_series(annually["emp"]).clip(-10,10)
        '''
        #########################
        ########### DE ##########
        #########################
    
             NA    ZERO    mean    std    skew     kurt
        DE  0.0  0.3923  0.0932  1.344  1.1086  34.2543
                     JB  99.5%  0.5%   min   max  median
        DE  744479.3618   10.0  -7.0 -10.0  10.0     0.0
        '''
        
        annually["RE"] = (((annually["fatl"].replace(np.nan,0)+annually["fatb"].replace(np.nan,0)-
                           annually["dpacb"].replace(np.nan,0))/\
            annually["at"]).replace([np.inf,-np.inf],0)).clip(0,1)
        '''
        #########################
        ########### RE ##########
        #########################
    
                NA    ZERO    mean     std    skew     kurt
        RE  0.1557  0.5172  0.0501  0.1193  3.9809  19.7254
                     JB   99.5%  0.5%  min  max  median
        RE  1649023.535  0.7644   0.0  0.0  1.0     0.0
        '''
        
        annually["SD"] = (annually["dm"].replace(np.nan,0)/(
            annually["dltt"].replace(np.nan,0)+annually["dlc"].replace(np.nan,0))).\
            replace([np.inf,-np.inf],0)
        '''
        #########################
        ########### SD ##########
        #########################
    
                NA    ZERO    mean     std    skew       kurt
        SD  0.2563  0.4256  0.2111  0.3495  8.8939  1027.4088
                      JB  99.5%  0.5%     min      max  median
        SD  2.759772e+07    1.0   0.0 -0.0001  48.7452     0.0
        '''
        annually = annually[["RE","SD","DE"]]
        
        quarterly = pd.merge(quarterly.reset_index(),annually, left_on = ["gvkey","datadate"],
                              right_on = ["gvkey","datadate"],how="left").set_index(["gvkey","date"])
        quarterly.sort_index(inplace=True)
        
        for _column in ["RE","SD","DE"]:
            quarterly[_column] = quarterly[_column].groupby("gvkey").ffill(limit=4)
        
        ### do we need to keep any? 
        ##### !!! check for duplidates here.
        ##### maybe check approach with dropping on FSS2 as earlier
        # quarterly = quarterly[~quarterly.index.duplicated(keep="first")] 
        
        quarterly[columns_quarterly_1].to_hdf(PATH_DATA+"quarterly_P3_s1.h5",key="data")
    
    
        ## quarterly elements of monthly kf
        quarterly_to_monthly_list = ["SP","CFP","EP","RDMCAP","BM"]
        
        quarterly[quarterly_to_monthly_list].to_hdf(PATH_DATA+"quarterly_to_monthly.h5",key="data")
    
    ##################################################
    ##################################################
    #############  Quarterly: Expansion  #############
    ##################################################
    ##################################################
    
    if not "SP" in quarterly.columns or "FULL" in mode:
        quarterly_signals = pd.read_hdf(PATH_DATA+"quarterly_P3_s1.h5","data")
        quarterly = pd.merge(quarterly,quarterly_signals,
                             left_index=True,right_index=True)
        columns_quarterly_1.extend(quarterly_signals.columns.to_list())
        del quarterly_signals
        quarterly_to_monthly = pd.read_hdf(PATH_DATA+"quarterly_to_monthly.h5","data")
        quarterly = pd.merge(quarterly,quarterly_to_monthly,
                             left_index=True,right_index=True)
        del quarterly_to_monthly
        
    ########################################
    ######### Additional Predictors ########
    ########################################
    columns_all = columns_quarterly_1 + []
    
    quarterly["DEAR"] = delta_series_relative(quarterly["EP"]).clip(-10,10)
    '''
    #########################
    ########## DEAR #########
    #########################

              NA    ZERO    mean     std    skew     kurt
    DEAR  0.1861  0.0193  0.0069  1.4956  0.1646  27.9059
                    JB  99.5%  0.5%   min   max  median
    DEAR  1.523092e+06   10.0 -10.0 -10.0  10.0  0.0179
    '''
    
    quarterly["RDE"] = (rolling_series(quarterly["redeq"])/rolling_series(quarterly["oiq"])).replace(
        [np.inf,-np.inf],np.nan).clip(-100,100)
    '''
    #########################
    ########## RDE ##########
    #########################

             NA    ZERO    mean      std    skew     kurt
    RDE  0.1694  0.1348  2.1301  17.1266  0.1656  19.7129
                   JB  99.5%   0.5%    min    max  median
    RDE  1.024594e+06  100.0 -100.0 -100.0  100.0  0.5418
    '''
    
    quarterly["RDC"] = (quarterly["cheq"]/rolling_series(quarterly["redeq"])).replace(
        [np.inf,-np.inf],np.nan).clip(0,100)
    '''
    #########################
    ########## RDC ##########
    #########################

             NA    ZERO    mean     std     skew       kurt
    RDC  0.3735  0.0067  0.4649  2.4255  29.7321  1090.0846
                   JB   99.5%  0.5%  min    max  median
    RDC  2.814919e+08  6.8612   0.0  0.0  100.0  0.1203
    '''
    
    quarterly["SGAE"] = (quarterly["EP"]/rolling_series(quarterly["xsgaq"].clip(lower=0))).replace(
        [np.inf,-np.inf],np.nan).clip(-100,100)
    '''
    #########################
    ########## SGAE #########
    #########################

              NA    ZERO    mean     std    skew      kurt
    SGAE  0.3357  0.0045  0.1695  3.4984  6.2443  394.9171
                    JB    99.5%    0.5%    min    max  median
    SGAE  3.335965e+07  11.8614 -8.4997 -100.0  100.0  0.1295
    '''
    
    quarterly["ARI"] = (quarterly["rectq"]/quarterly["invtq"]).\
        replace([np.inf,-np.inf],np.nan).clip(0,100)
    '''
    #########################
    ########## ARI ##########
    #########################

             NA    ZERO     mean      std    skew    kurt
    ARI  0.4894  0.0047  11.0221  26.3433  2.8132  6.3986
                   JB  99.5%  0.5%  min    max  median
    ARI  2.134391e+06  100.0   0.0  0.0  100.0   1.265
    '''
    
    quarterly["DSGAE"] = delta_series_relative(quarterly["oiq"]+quarterly["xsgaq"]).clip(-100,100)
    '''
    #########################
    ######### DSGAE #########
    #########################

               NA    ZERO    mean     std    skew      kurt
    DSGAE  0.4126  0.0003 -0.0337  4.5442 -1.5558  324.4866
                     JB   99.5%   0.5%    min    max  median
    DSGAE  2.016428e+07  9.5761 -10.68 -100.0  100.0  0.0215
    '''
    
    quarterly["CFRD"] = (quarterly["CFP"]/quarterly["redeq"]).replace(
        [np.inf,-np.inf],np.nan).clip(-100,100)
    '''
    #########################
    ########## CFRD #########
    #########################

              NA    ZERO    mean    std    skew      kurt
    CFRD  0.3013  0.0003  0.4918  5.037  4.3763  197.0184
                    JB    99.5%     0.5%    min    max  median
    CFRD  1.647805e+07  20.7935 -11.5396 -100.0  100.0  0.1436
    '''
    
    ########################################
    ######## Predictors Derivatives ########
    ########################################
    
    quarterly["DACA"] = delta_series_relative(quarterly["ACA"]).clip(-10,10)
    '''
    #########################
    ########## DACA #########
    #########################

             NA   ZERO    mean     std    skew    kurt
    DACA  0.306  0.001 -1.2008  2.9879 -1.9775  2.8768
                   JB  99.5%  0.5%   min  max  median
    DACA  944917.9275    1.0 -10.0 -10.0  1.0  0.0232
    '''
    
    quarterly["DROI"] = delta_series_relative(quarterly["ROI"]).clip(-10,10)
    '''
    #########################
    ########## DROI #########
    #########################

              NA   ZERO    mean     std    skew     kurt
    DROI  0.3799  0.005 -0.0127  1.1759  0.2634  45.1968
                    JB   99.5%    0.5%   min   max  median
    DROI  2.586188e+06  7.2265 -6.6063 -10.0  10.0     0.0
    '''
    
    quarterly["DROE"] = delta_series_relative(quarterly["ROE"]).clip(-10,10)
    '''
    #########################
    ########## DROE #########
    #########################

              NA    ZERO   mean     std    skew     kurt
    DROE  0.2945  0.0008 -0.043  1.7002  0.0659  20.9748
                    JB  99.5%  0.5%   min   max  median
    DROE  1.095512e+06   10.0 -10.0 -10.0  10.0 -0.0099
    '''
    
    quarterly["DRDS"] = delta_series_relative(quarterly["RDS"]).clip(-10,10)
    '''
    #########################
    ########## DRDS #########
    #########################

              NA    ZERO    mean     std     skew     kurt
    DRDS  0.3328  0.0192 -0.0161  0.3388 -16.6224  457.614
                    JB  99.5%    0.5%   min   max  median
    DRDS  9.497538e+07    1.0 -1.0921 -10.0  10.0     0.0
    '''
    
    quarterly["DFSS"] = delta_series(quarterly["FSS"])
    '''
    #########################
    ########## DFSS #########
    #########################

           NA   ZERO    mean    std    skew    kurt
    DFSS  0.0  0.471  0.0342  1.339  0.0747  2.0536
                  JB  99.5%  0.5%  min  max  median
    DFSS -56267.5611    4.0  -4.0 -9.0  8.0     0.0
    '''
    
    columns_quarterly_2 = columns_quarterly_1+[
        "DEAR", "RDE", "RDC", "SGAE", "ARI", "DSGAE", "CFRD", "DACA", "DROI", "DROE", "DRDS", "DFSS"
        ]
    quarterly = quarterly[~quarterly.index.duplicated(keep="first")]
    
    quarterly[columns_quarterly_2].to_hdf(PATH_DATA+"quarterly_P3_s2.h5",key="data")
    
####################################################################################################
####################################################################################################
#######################################  Weekly (from CRSP)  #######################################
####################################################################################################
####################################################################################################
'''
Weekly data for algorithm

Problem: Input file is quite large (8gb).
Solution:
    split input file in subfiles.
    
    
Step1:
    calculate daily market capitalisations and returns
Step2:
    calculate historical market index
    
'''

def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False

def numbermaker(nr,std = 0):
    if isnumber(nr):
        return float(np.nan_to_num(nr))
    return std

def groupby_beta():
    '''
    This function should computationally efficiently calculate the beta between the market and 
    individual securities.
    '''
    pass

if ("FULL" in mode or "mcap" in mode):
    
    ##################################################
    ##################################################
    ##############  Weekly: Daily MCAP  ##############
    ##################################################
    ##################################################
    
    skiprow, nrow = 0,5000000
    ## import matching table
    matching_table = pd.read_hdf(PATH_DATA+"CRSP_Compustat_match.h5","data")
    market_cap_returns = []
    stopping_criterion = False
    nobs_total, nobs_matched = 0, 0
    
    ## load report dates
    report_dates = pd.read_hdf(PATH_DATA+"report_dates.h5","data")
    report_dates["rdq"] = report_dates["rdq"].astype(str)
    report_dates["rdq"] = report_dates["rdq"].apply(lambda x: dt.datetime.strptime(x, "%Y%m%d"))
    del report_dates["cusip"]
    report_dates["date"] = report_dates["rdq"]
    report_dates.drop_duplicates(["gvkey", "date"],inplace=True)
    ## earnings announcement kf
    ea = []
    monthly_rvar = []
    
    while not stopping_criterion:
        # skiprow = 0; nrow = 5000000
        ## import part of the daily file
        daily = pd.read_csv(PATH_DATA+"daily.csv",
                            skiprows=np.arange(1,skiprow),nrows=nrow)
        
        if daily.shape[0]!= nrow:
            stopping_criterion = True
            
        
        ## delete incomplete PERMNO from end of daily file
        daily = daily[daily["PERMNO"] != daily["PERMNO"].values[-1]]
        del daily["PERMNO"]
        skiprow+=daily.shape[0]
        
        nobs_total +=daily.shape[0]
        
        ## Use only companies with quarterly match
        daily = pd.merge(matching_table[["gvkey","CUSIP"]],daily,
                         left_on ="CUSIP",right_on = "CUSIP",how="inner")
        nobs_matched +=daily.shape[0]
        
        del daily["CUSIP"]
        daily.dropna(subset= ["PRC","SHROUT"],inplace=True)
        daily.drop_duplicates(subset=["gvkey","date"],keep="first")
        for column in ["RET","DLRET","PRC","SHROUT","VOL"]:
            daily[column] = pd.to_numeric(
                daily[column],errors= "coerce").fillna(0)
        daily.set_index(["gvkey","date"],inplace=True)
        daily.loc[daily["RET"].isin([-66,-77,-88,-99]),"RET"] = 0
        # daily["DLRET"] = pd.to_numeric(daily["DLRET"],errors= "coerce").fillna(0)
        daily.loc[daily["DLRET"].isin([-66,-77,-88,-99]),"DLRET"] = 0
        daily["RET"] = ((1+daily["RET"])*(1+daily["DLRET"])-1).clip(-.99,10)
        daily["VOL"] = daily["VOL"].replace(np.nan,0)
        
        market_cap_returns.append(pd.DataFrame({
            "MCAP": daily["PRC"].abs()*daily["SHROUT"],
            "ret_d": daily["RET"]}))
        
        
        #### date preprocessing
        daily.reset_index(inplace=True)
        daily["date"] = daily["date"].apply(lambda x: dt.datetime.strptime(x,"%Y-%m-%d"))
        
        gvkeys_local = daily["gvkey"].unique().tolist()
        report_dates_local = report_dates[report_dates["gvkey"].isin(gvkeys_local)]
        daily = pd.merge(daily,report_dates_local,left_on = ["gvkey","date"],
                         right_on = ["gvkey","date"],how="outer")
        daily.sort_values(["gvkey","date"],inplace=True)
        daily["rdq_pre"] = daily["rdq"]
        daily["rdq_pre"] = daily.groupby("gvkey")["rdq_pre"].bfill(limit = 61)
        daily["rdq_post"] = daily["rdq"]
        daily["rdq_post"] = daily.groupby("gvkey")["rdq_post"].bfill(limit = 1)
        daily["rdq_post"] = daily.groupby("gvkey")["rdq_post"].ffill(limit = 1)
        daily.loc[daily["rdq_post"]==daily["rdq_pre"],"rdq_pre"] = np.nan
        
        rdq_pre = daily.loc[~daily["rdq_pre"].isna(),["rdq_pre","rdq","gvkey","RET","VOL"]].groupby(["rdq_pre","gvkey"])
        rdq_post = daily.loc[~daily["rdq_post"].isna(),["rdq_post","date","rdq","gvkey","RET","VOL"]].groupby(["rdq_post","gvkey"])
        ea_pre = pd.DataFrame({
            "ear":rdq_pre["RET"].mean(),
            "eav":rdq_pre["VOL"].mean()
            })
        del rdq_pre
        ea_pre.index.set_names({"rdq_pre":"rdq_post"},inplace=True)
        ea_local = pd.DataFrame({
            "EAR":rdq_post["RET"].mean()-ea_pre["ear"],
            "EAV":rdq_post["VOL"].mean()/ea_pre["eav"],
            "date":rdq_post["date"].last(),
            "gvkey":rdq_post["gvkey"].first(),
            })
        ea_local["EAR"] = ea_local["EAR"].clip(-2,12)
        ea_local["EAV"] = ea_local["EAV"].clip(0,10)
        ea_local.set_index(["gvkey","date"],drop=True,inplace=True)
        ea.append(ea_local[(~ea_local["EAV"].isna())|(~ea_local["EAR"].isna())])
        
        daily["month"] = daily["date"].dt.month
        daily["year"] = daily["date"].dt.year
        
        daily.loc[:,"RET"] = np.log(daily["RET"]+1)
        daily.loc[:,"RET2"] = daily.loc[:,"RET"]**2
    
        groupby_daily = daily.groupby(["gvkey","year","month"])
        monthly_add = pd.DataFrame({
            "RVAR1M":groupby_daily["RET2"].sum(),
            "count_month": groupby_daily["RET"].count(),
            "date":groupby_daily["date"].max(),
            "gvkey":groupby_daily["gvkey"].first()
                })
        
        monthly_add['date'] = pd.to_datetime(monthly_add['date'], format="%Y%m") + MonthEnd(0)
        
        monthly_add["RVAR6M"] =  monthly_add["RVAR1M"].groupby("gvkey").rolling(6,min_periods=2).sum().droplevel(0)/\
            monthly_add["count_month"].groupby("gvkey").rolling(6,min_periods=2).sum().droplevel(0)
        monthly_add["RVAR1M"] = monthly_add["RVAR1M"]/monthly_add["count_month"]
        
        monthly_rvar.append(monthly_add)
        del daily
    
    ea = pd.concat(ea)
    ea.to_hdf(PATH_DATA+"earnings_announcement_raw.h5",key="data")

    monthly_rvar = pd.concat(monthly_rvar)
    monthly_rvar.set_index(["gvkey","date"])[["RVAR6M","RVAR1M"]].to_hdf("monthly_rvar.h5",key="data")
        
        
    
    '''
    ##########   Summary Statistics for CRSP daily observations   #########
    
    Matched CRSP     :  61250744; Unmatched CRSP     :   33533745 % matched: 64.62%
    '''
    
    
    market_cap_returns = pd.concat(market_cap_returns)
    
    ## save market cap values daily
    market_cap_returns["MCAP"].to_hdf(PATH_DATA+"daily_MCAP.h5",key="data")
    
    
    
    ##################################################
    ##################################################
    ##########  Weekly: Daily Market Return  #########
    ##################################################
    ##################################################
    
    market_cap_returns["ret_d"] = market_cap_returns["ret_d"]
    ## create market portfolio
    market_cap_returns["MCAP1"] = market_cap_returns["MCAP"].groupby("gvkey").shift(1)
    market_cap_returns["MCAP1"].fillna(0)
    market_return = pd.DataFrame({
        "raw_ret" : (market_cap_returns["ret_d"]*market_cap_returns["MCAP1"]).groupby("date").sum(),
        "raw_mcap": market_cap_returns["MCAP1"].groupby("date").sum()
        })
    
    market_return["MPV"] = (market_return["raw_ret"]/market_return["raw_mcap"]).fillna(0)
    
    market_return["MPE"] = market_cap_returns["ret_d"].groupby("date").mean()
    
    ########################################
    ######### Market Excess returns ########
    ########################################
    
    ## load tbill rates
    ustb3d = pd.read_csv(PATH_DATA+"USTB3M.csv").rename(
        columns={"DATE":"date"}).set_index("date")
    
    ustb3d["DTB3"] = pd.to_numeric(
        ustb3d["DTB3"],errors= "coerce").fillna(0)
    
    ustb3d["DTB3"] =(1+ustb3d["DTB3"]/100)**(1/252)-1
    
    market_return = pd.merge(market_return, ustb3d, left_index=True, right_index= True,
                             how="left")
    market_return["MPEe"] = market_return["MPE"]-ustb3d["DTB3"]
    market_return["MPVe"] = market_return["MPV"]-ustb3d["DTB3"]
    
    ## save market portfolio:
    market_return[["MPV","MPE","MPVe","MPEe"]].to_csv(PATH_DATA+"daily_market.csv")

if ("FULL" in mode or "weekly" in mode):
    ##################################################
    ##################################################
    ################  Weekly: Signals  ###############
    ##################################################
    ##################################################
    stopping_criterion = False
    
    ## create iterators for iterative saving process of weekly kf.
    skiprow, nrow = 0,5000000
    print_fmt("weekly calculation")
    print("Stepsize",nrow,end = "\n\n")
    ##################################
    #  Loading of additional tables  #
    ##################################
    
    ## load matching table CRSP_Compustat
    matching_table = pd.read_hdf(PATH_DATA+"CRSP_Compustat_match.h5","data")
    
    ## load tbill rates
    ustb3m = pd.read_csv(PATH_DATA+"USTB3M.csv").rename(
        columns={"DATE":"date"})
    ustb3m["DTB3"] = pd.to_numeric(
        ustb3m["DTB3"],errors= "coerce").fillna(0)
    ustb3m["DTB3"] =(1+ustb3m["DTB3"]/100)**(1/252)-1
    
    ustb3m["date"] = ustb3m["date"].apply(lambda x: dt.datetime.strptime(x,"%Y-%m-%d"))
    ustb3m["week"] = ustb3m["date"].dt.isocalendar().week
    ustb3m["year"] = ustb3m["date"].dt.isocalendar().year
    ustb3m["DTB3"] += 1
    ustb3m_weekly = pd.DataFrame({
        "date": ustb3m.groupby(["year","week"])["date"].max(),
        "WTB3": (ustb3m.groupby(["year","week"])["DTB3"]).prod()-1,
        "week": ustb3m.groupby(["year","week"])["week"].first(),
        "year": ustb3m.groupby(["year","week"])["year"].first(),
        }).set_index("date",drop=True)
    ustb3m["DTB3"] -= 1
    
    
    del ustb3m["week"], ustb3m["year"]
    ustb3m.set_index("date",inplace=True)
    
    ## load market excess return for beta factor 
    market = pd.read_csv(PATH_DATA+"daily_market.csv")[["date","MPVe"]]
    
    market["date"] = market["date"].apply(lambda x: dt.datetime.strptime(x,"%Y-%m-%d"))
    market["week"] = market["date"].dt.isocalendar().week
    market["year"] = market["date"].dt.isocalendar().year
    market["MPVe"] += 1
    market_weekly = pd.DataFrame({
        "date": market.groupby(["year","week"])["date"].max(),
        "MPVe": (market.groupby(["year","week"])["MPVe"]).prod()-1,
        "week": market.groupby(["year","week"])["week"].first(),
        "year": market.groupby(["year","week"])["year"].first(),
        }).set_index("date",drop=True)
    market["MPVe"] -= 1
    
    
    del market["week"], market["year"]
    market.set_index("date",inplace=True)
    
    weekly = []
    weekly_ind = []
    
    ## iterate over all files loading
    while not stopping_criterion:
        # skiprow, nrow = 0, 5000000
        print("Calculation of weekly data at {rows:d}.".format(rows=skiprow))
        daily = pd.read_csv(PATH_DATA+"daily.csv",
                            skiprows=np.arange(1,skiprow),nrows=nrow)
        
        
        if daily.shape[0]!= nrow:
            stopping_criterion = True
            
        ## delete incomplete PERMNO from end of daily file
        daily = daily[daily["PERMNO"] != daily["PERMNO"].values[-1]]
        del daily["PERMNO"]
        skiprow+=daily.shape[0]
        
        ## Use only companies with quarterly match
        daily = pd.merge(matching_table[["gvkey","CUSIP"]],daily,
                         left_on ="CUSIP",right_on = "CUSIP",how="inner")
        
        #### preprocessing the relevant columns
        del daily["CUSIP"]
        daily.dropna(subset= ["PRC","SHROUT"],inplace=True)
        daily.drop_duplicates(subset=["gvkey","date"],keep="first")
        for column in ["RET","DLRET","PRC","SHROUT"]:
            daily[column] = pd.to_numeric(
                daily[column],errors= "coerce").fillna(0)
        daily.set_index(["gvkey","date"],inplace=True)
        daily.loc[daily["RET"].isin([-66,-77,-88,-99]),"RET"] = 0
        # daily["DLRET"] = pd.to_numeric(daily["DLRET"],errors= "coerce").fillna(0)
        daily.loc[daily["DLRET"].isin([-66,-77,-88,-99]),"DLRET"] = 0
        daily["RET"] = ((1+daily["RET"])*(1+daily["DLRET"])-1).clip(-.99,10)
        
        #### date preprocessing
        daily.reset_index(inplace=True)
        daily["date"] = daily["date"].apply(lambda x: dt.datetime.strptime(x,"%Y-%m-%d"))
        daily["week"] = daily["date"].dt.isocalendar().week
        daily["year"] = daily["date"].dt.isocalendar().year
        # ## Uncommented: calculation of excess returns also for volatility
        # daily = pd.merge(daily,ustb3m,left_on = "date",right_on = "date",how="left")
        # daily["RET"]-=daily["DTB3"]
        
        ## logarithmise for ease of calculation
        daily.loc[:,"RET"] = np.log(daily["RET"]+1)
        daily.loc[:,"RET2"] = daily.loc[:,"RET"]**2
        daily.loc[:,"QUART"] = daily.loc[:,"RET"]**4
        
        ## trading volume and zero trading
        daily["VOL"] = daily["VOL"].replace(np.nan,0)
        daily["ZTD"] = 0
        daily.loc[(daily["VOL"]==0)&(daily["RET"]==0),"ZTD"] = 1
        
        ## dollar trading volume
        daily["dolvol"] = daily["PRC"].abs()*daily["SHROUT"]
        daily["dolvol2"] = daily["dolvol"]**2
        
        ## MCAP
        daily["MCAP"] = daily["PRC"].abs()*daily["SHROUT"]
        
        ## BAS
        daily["BAS"] = (daily["ASKHI"].abs()-daily["BIDLO"].abs())/daily["BIDLO"].abs()
        
        ## absolute daily price movement, number of trades
        daily["DPRC"] = daily["RET"].abs()
        
        ## SICCD
        daily["SICCD"] = daily.groupby("gvkey")["gvkey"].ffill()
        daily["SICCD"] = daily.groupby("gvkey")["gvkey"].bfill()
        daily["VOL2"] = daily["VOL"]**2
        
        groupby_daily = daily.groupby(["gvkey","year","week"])
        weekly_add = pd.DataFrame({
            "date":groupby_daily["date"].max(),
            "m1w":groupby_daily["RET"].sum(),
            "rvar1w":groupby_daily["RET2"].sum(),
            "rq":groupby_daily["QUART"].sum(),
            "gvkey":groupby_daily["gvkey"].first(), 
            "SICCD":groupby_daily["SICCD"].first(),
            "mdr1w":groupby_daily["RET"].max(),
            "ztd":groupby_daily["ZTD"].sum(),
            "count1w":groupby_daily["ZTD"].count(),
            "mcap":np.log(groupby_daily["MCAP"].last()),
            "bas":groupby_daily["BAS"].sum(),
            "shrout":groupby_daily["SHROUT"].last(),
            "vol":groupby_daily["VOL"].sum(),
            "stdvol":groupby_daily["VOL"].std(),
            "vol2":groupby_daily["VOL2"].sum(),
            "dolvol":groupby_daily["dolvol"].sum(),
            "dolvol2":groupby_daily["dolvol2"].sum(),
            "stddolvol":groupby_daily["dolvol"].std(),
            "ILL_absdolmov":groupby_daily["DPRC"].sum(),
            "ILL_numtrd":groupby_daily["NUMTRD"].sum()
            })
        del groupby_daily
        
        ## apply treasury bill rate, logaritithmise for groupby commands
        weekly_add              = pd.merge(weekly_add,ustb3m_weekly,
                                           left_on = ["year","week"],
                                           right_on = ["year","week"],how="left")
        weekly_add              = pd.merge(weekly_add,market_weekly,
                                           left_on = ["year","week"],
                                           right_on = ["year","week"],how="left")
        
        
        weekly_add.loc[:,"m1w"]  = np.log(np.exp(weekly_add.loc[:,"m1w"])-weekly_add["WTB3"])
        del weekly_add["WTB3"]
        weekly_add.loc[:,"m1w"] = weekly_add.loc[:,"m1w"].replace(np.nan,np.log(.01))
        weekly_add.loc[:,"m1w"] = weekly_add.loc[:,"m1w"].clip(np.log(.01),np.log(11))
        # weekly_add.loc[:,"m1w"]   = np.log(weekly_add.loc[:,"m1w"])
        """
        #########################
        ########## m1w ##########
        #########################

              NA  ZERO    mean    std    skew      kurt
        m1w  0.0   0.0  0.0018  0.085  7.2471  412.6777
                       JB  99.5%    0.5%   min   max  median
        m1w  3.219978e+08  0.349 -0.2605 -0.99  10.0  -0.001
        """
        
        ## index
        weekly_add.set_index(["gvkey","date"],inplace=True)
        weekly_add.sort_index(inplace=True)
        
        #### calculate momentum factors and returns
        ## momentum factors
        weekly_add["m4w"]         = weekly_add.groupby("gvkey")["m1w"].rolling(4,min_periods=2).sum().\
            droplevel(0) - weekly_add["m1w"] 
        """
        #########################
        ########## m4w ##########
        #########################

                NA  ZERO    mean     std    skew       kurt
        m4w  0.002   0.0  0.0049  0.1433  8.9732  1042.0155
                       JB   99.5%    0.5%     min      max  median
        m4w  7.071572e+08  0.5999 -0.4073 -0.9982  43.4724 -0.0008
        """
        
        weekly_add["m13w"]        = weekly_add.groupby("gvkey")["m1w"].rolling(13,min_periods=5).sum().\
            droplevel(0) - weekly_add["m1w"] - weekly_add["m4w"]
        """
        #########################
        ########## m13w #########
        #########################

                  NA  ZERO    mean     std    skew      kurt
        m13w  0.0078   0.0  0.0144  0.2507  8.3204  522.0697
                        JB   99.5%    0.5%     min      max  median
        m13w  4.135573e+08  1.0856 -0.6038 -0.9998  43.9905  0.0027
        """
        
        weekly_add["m26w"]        = weekly_add.groupby("gvkey")["m1w"].rolling(26,min_periods=14).sum().\
            droplevel(0) - weekly_add["m1w"] - weekly_add["m4w"] - weekly_add["m13w"]
        """
       #########################
       ########## m26w #########
       #########################

                 NA  ZERO   mean     std    skew      kurt
       m26w  0.0253   0.0  0.021  0.3066  9.8791  625.1463
                       JB  99.5%    0.5%     min      max  median
       m26w  5.260634e+08  1.344 -0.6625 -0.9999  53.7448  0.0046
        """
        
        
        ## future returns
        weekly_add["ret1w"] = weekly_add["m1w"].groupby("gvkey").shift(-1)
        """
        #########################
        ######### ret1w #########
        #########################

                  NA  ZERO    mean     std    skew      kurt
        ret1w  0.002   0.0  0.0018  0.0849  7.2066  411.2849
                         JB   99.5%    0.5%   min   max  median
        ret1w  3.200582e+08  0.3489 -0.2605 -0.99  10.0  -0.001
        """
    
        weekly_add["ret4w"] = weekly_add["ret1w"].groupby("gvkey").rolling(4,min_periods=2).sum().\
            groupby("gvkey").shift(-3).droplevel(0)
        """
        #########################
        ######### ret4w #########
        #########################

                   NA  ZERO    mean     std    skew      kurt
        ret4w  0.0058   0.0  0.0063  0.1649  6.0142  237.8463
                         JB   99.5%   0.5%     min      max  median
        ret4w  1.971862e+08  0.6989 -0.462 -0.9993  18.4498    -0.0
        """
        
        weekly_add["ret13w"] = weekly_add["ret1w"].groupby("gvkey").rolling(13,min_periods=5).sum().\
            groupby("gvkey").shift(-12).droplevel(0)
        """
        #########################
        ######### ret13w ########
        #########################

                    NA  ZERO    mean     std    skew      kurt
        ret13w  0.0233   0.0  0.0207  0.3105  9.3588  562.9184
                          JB   99.5%   0.5%     min      max  median
        ret13w  4.729343e+08  1.3597 -0.684 -0.9999  53.7448   0.005
        """
        
        
        ## taking out the logarithm
        for column in ["m1w","m4w","m13w","m26w","ret1w","ret4w","ret13w"]:
            weekly_add[column] = np.exp(weekly_add[column])-1
        
        #### calculate realized volatility and relaized volatility forward
        weekly_add["rvar4w"]    = weekly_add["rvar1w"].groupby("gvkey").rolling(4,min_periods=2).\
            sum().droplevel(0)/weekly_add["count1w"].groupby("gvkey").rolling(4,min_periods=2).\
                sum().droplevel(0)
        """
        #########################
        ######### rvar4w ########
        #########################

                   NA    ZERO    mean     std     skew      kurt
        rvar4w  0.002  0.0058  0.0016  0.0047  11.2598  182.1534
                          JB   99.5%  0.5%  min  max  median
        rvar4w  3.565587e+08  0.0277   0.0  0.0  0.1  0.0005
        """
        
        weekly_add["rvar13w"]   = weekly_add["rvar1w"].groupby("gvkey").rolling(13,min_periods=5).\
            sum().droplevel(0)/weekly_add["count1w"].groupby("gvkey").rolling(13,min_periods=5).\
                sum().droplevel(0)
        """
        #########################
        ######## rvar13w ########
        #########################

                     NA   ZERO    mean     std     skew      kurt
        rvar13w  0.0078  0.001  0.0016  0.0039  10.6685  186.4919
                           JB   99.5%  0.5%  min  max  median
        rvar13w  3.318700e+08  0.0231   0.0  0.0  0.1  0.0006
        """
        
        weekly_add["rvar26w"]   = weekly_add["rvar1w"].groupby("gvkey").rolling(26,min_periods=14).\
            sum().droplevel(0)/weekly_add["count1w"].groupby("gvkey").rolling(26,min_periods=14).\
                sum().droplevel(0)
        """
        #########################
        ######## rvar26w ########
        #########################

                     NA    ZERO    mean     std    skew      kurt
        rvar26w  0.0253  0.0003  0.0016  0.0034  9.9135  181.3508
                           JB   99.5%  0.5%  min  max  median
        rvar26w  2.969042e+08  0.0201   0.0  0.0  0.1  0.0006
        """
        
        
        weekly_add["rvarf1w"] = weekly_add["rvar1w"].groupby("gvkey").shift(-1)
        weekly_add["countf1w"] = weekly_add["count1w"].groupby("gvkey").shift(-1)
        weekly_add["rvarf4w"] = weekly_add["rvar1w"].groupby("gvkey").rolling(4,min_periods=2).sum().\
            groupby("gvkey").shift(-4).droplevel(0)/weekly_add["count1w"].groupby("gvkey").rolling(4,min_periods=2).sum().\
                groupby("gvkey").shift(-4).droplevel(0)
        """
        #########################
        ######## rvarf4w ########
        #########################

                     NA    ZERO    mean     std     skew      kurt
        rvarf4w  0.0078  0.0057  0.0016  0.0046  11.2532  182.2134
                           JB   99.5%  0.5%  min  max  median
        rvarf4w  3.562833e+08  0.0276   0.0  0.0  0.1  0.0005
        """
        
        weekly_add["rvarf13w"] = weekly_add["rvar1w"].groupby("gvkey").rolling(13,min_periods=5).sum().\
            groupby("gvkey").shift(-13).droplevel(0)/weekly_add["count1w"].groupby("gvkey").rolling(13,min_periods=5).sum().\
                groupby("gvkey").shift(-13).droplevel(0)
        """
        #########################
        ######## rvarf13w #######
        #########################

                      NA   ZERO    mean     std     skew      kurt
        rvarf13w  0.0253  0.001  0.0016  0.0039  10.5921  184.4933
                            JB   99.5%  0.5%  min  max  median
        rvarf13w  3.274529e+08  0.0231   0.0  0.0  0.1  0.0006
        """
        
        weekly_add.loc[:,"rvarf1w"] = weekly_add["rvarf1w"]/weekly_add["countf1w"]
        """
        #########################
        ######## rvarf1w ########
        #########################

                    NA    ZERO    mean     std     skew     kurt
        rvarf1w  0.002  0.0247  0.0016  0.0054  11.1499  164.515
                           JB   99.5%  0.5%  min  max  median
        rvarf1w  3.422769e+08  0.0331   0.0  0.0  0.1  0.0004
        """

        del weekly_add["countf1w"]
        
        weekly_add.loc[:,"rvar1w"] = weekly_add["rvar1w"]/weekly_add["count1w"]
        """
        #########################
        ######### rvar1w ########
        #########################

                 NA    ZERO    mean     std    skew      kurt
        rvar1w  0.0  0.0253  0.0016  0.0054  11.149  164.4291
                          JB   99.5%  0.5%  min  max  median
        rvar1w  3.421923e+08  0.0331   0.0  0.0  0.1  0.0003
        """
        ## Realised quarticitiy following Bollerslev et. al. (2016)
        weekly_add.loc[:,"rq1w"] = weekly_add["rq"]*weekly_add["count1w"]/3
        '''
        #########################
        ########## rq1w #########
        #########################

               NA    ZERO    mean     std     skew      kurt
        rq1w  0.0  0.0253  0.0008  0.0141  29.5894  963.8113
                        JB   99.5%  0.5%  min  max  median
        rq1w  2.318734e+09  0.0221   0.0  0.0  0.5     0.0
        '''
        
        weekly_add.loc[:,"rq4w"] = weekly_add["rq"].groupby("gvkey").rolling(4,min_periods=2).\
            sum().droplevel(0)*weekly_add["count1w"].groupby("gvkey").rolling(4,min_periods=2).\
                sum().droplevel(0)/3
        '''
        #########################
        ########## rq4w #########
        #########################

                 NA    ZERO    mean     std    skew     kurt
        rq4w  0.002  0.0058  0.0076  0.0447  9.1952  91.0397
                        JB  99.5%  0.5%  min  max  median
        rq4w  2.214556e+08    0.5   0.0  0.0  0.5  0.0001
        '''        
                
        weekly_add.loc[:,"rq13w"] = weekly_add["rq"].groupby("gvkey").rolling(13,min_periods=5).\
            sum().droplevel(0)*weekly_add["count1w"].groupby("gvkey").rolling(13,min_periods=5).\
                sum().droplevel(0)/3
        '''
        #########################
        ######### rq13w #########
        #########################

                   NA   ZERO    mean     std    skew     kurt
        rq13w  0.0078  0.001  0.0413  0.1088  3.3914  10.6908
                         JB  99.5%  0.5%  min  max  median
        rq13w  2.789837e+07    0.5   0.0  0.0  0.5  0.0021
        '''
        
        
        ## Maximum daily return previous 4 weeks
        weekly_add["mdr4w"] = weekly_add["mdr1w"].groupby("gvkey").rolling(4).max().droplevel(0)
        '''
        #########################
        ######### mdr1w #########
        #########################

                NA    ZERO    mean     std    skew      kurt
        mdr1w  0.0  0.0721  0.0334  0.0473  2.6004  367.2556
                         JB   99.5%    0.5%     min     max  median
        mdr1w  2.033032e+08  0.2719 -0.0093 -4.6052  2.3979  0.0205
        
        #########################
        ######### mdr4w #########
        #########################

                   NA    ZERO    mean     std    skew     kurt
        mdr4w  0.0058  0.0132  0.0623  0.0688  5.0865  59.2613
                         JB   99.5%  0.5%     min     max  median
        mdr4w  8.300028e+07  0.4055   0.0 -0.0054  2.3979   0.043
        '''
        #########################
        # Weekly Predictors old #
        #########################
        min_periods = 4
        n_periods =  26
        
        weekly_add["_prod_"] = weekly_add[["m1w","MPVe"]].prod(axis=1)
        groupby_obj = weekly_add[["m1w","MPVe","_prod_"]].groupby("gvkey")
        weekly_add["b"]		    = (groupby_obj["_prod_"].apply(lambda x: x.rolling(
            window=n_periods,min_periods=min_periods).mean())-(
            groupby_obj["m1w"].apply(lambda x: x.rolling(window=n_periods,min_periods=min_periods).mean())*\
            groupby_obj["MPVe"].apply(lambda x: x.rolling(window=n_periods,min_periods=min_periods).mean()))).droplevel(0)/\
            groupby_obj["MPVe"].apply(lambda x: x.rolling(window=n_periods,min_periods=min_periods).std()**2).droplevel(0)
        del groupby_obj, weekly_add["_prod_"]
        weekly_add["b"] = weekly_add["b"].clip(-5,5)
        '''
        26 week beta with the market

        #########################
        ########### b ###########
        #########################

               NA  ZERO    mean     std    skew    kurt
        b  0.0058   0.0  0.8071  1.0185  0.0786  4.2091
                    JB   99.5%    0.5%  min  max  median
        b  641024.1789  4.6761 -2.8248 -5.0  5.0  0.7519
        '''

        weekly_add["b2"]        = weekly_add["b"]**2
        '''
        Squared 26 week beta with the market.

        #########################
        ########### b2 ##########
        #########################

                NA  ZERO    mean     std    skew     kurt
        b2  0.0058   0.0  1.6888  3.1069  4.3103  23.6968
                      JB    99.5%  0.5%  min   max  median
        b2  4.936331e+07  24.9363   0.0  0.0  25.0  0.6705
        '''
        

        weekly_add["bas1w"]		= (weekly_add["bas"]/weekly_add["count1w"]).clip(0,2)
        '''
        Bid-Ask Spread 1 week
        
        #########################
        ######### bas1w #########
        #########################

                NA    ZERO    mean     std     skew      kurt
        bas1w  0.0  0.0014  0.0529  0.1005  10.2599  153.5353
                         JB   99.5%   0.5%  min  max  median
        bas1w  2.969747e+08  0.6228  0.001  0.0  2.0  0.0303
        '''
        
        weekly_add["ztd1w"]     = weekly_add["ztd"]/weekly_add["count1w"]
        '''
        Number of zero trading days.

        #########################
        ######### ztd1w #########
        #########################

                NA    ZERO    mean     std    skew     kurt
        ztd1w  0.0  0.9121  0.0444  0.1662  4.2304  17.9858
                         JB  99.5%  0.5%  min  max  median
        ztd1w  4.497791e+07    1.0   0.0  0.0  1.0     0.0
        '''

        weekly_add["tv1w"]		= np.log(weekly_add["dolvol"]+1)
        '''
        Log dollar trading volume
        
        #########################
        ########## tv1w #########
        #########################

               NA  ZERO     mean     std    skew    kurt
        tv1w  0.0   0.0  13.5438  2.2045  0.2258 -0.0703
                        JB    99.5%    0.5%     min     max  median
        tv1w -1.489249e+06  19.5325  8.5071  2.9144  23.406  13.456
        '''
        weekly_add["tv4w"]		= np.log(weekly_add["dolvol"].groupby("gvkey").rolling(4,min_periods=2).\
                                      sum().droplevel(0)/4+1)
        '''
        #########################
        ########## tv4w #########
        #########################

                 NA  ZERO    mean     std    skew    kurt
        tv4w  0.002   0.0  13.549  2.1996  0.2316 -0.0741
                        JB    99.5%    0.5%     min      max   median
        tv4w -1.485728e+06  19.5297  8.5469  3.1426  23.3517  13.4592
        '''

        weekly_add["vst1w"]		= (weekly_add["stdvol"]/weekly_add["shrout"]).clip(0,50)
        '''
        std log dollar trading volume
        
        #########################
        ######### vst1w #########
        #########################

                   NA    ZERO    mean     std    skew     kurt
        vst1w  0.0012  0.0416  3.9098  7.9124  4.1786  19.1425
                         JB  99.5%  0.5%  min   max  median
        vst1w  4.467473e+07   50.0   0.0  0.0  50.0  1.3774
        '''
        weekly_add["vst4w"] = (((weekly_add["vol2"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0)/\
            weekly_add["count1w"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0))-\
            (weekly_add["vol"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0)/\
             weekly_add["count1w"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0))**2)**.5/\
            weekly_add["shrout"]).clip(0,50)
        '''
        #########################
        ######### vst4w #########
        #########################

                  NA    ZERO    mean    std    skew     kurt
        vst4w  0.002  0.0337  5.3343  9.231  3.3949  12.1509
                         JB  99.5%  0.5%  min   max  median
        vst4w  2.870720e+07   50.0   0.0  0.0  50.0  2.1378
        '''
            
        
        weekly_add["sto1w"]		= (weekly_add["vol"]/weekly_add["shrout"]).clip(0,50)
        '''
        relative turnover of shares
        
        #########################
        ######### sto1w #########
        #########################

                NA    ZERO     mean      std    skew    kurt
        sto1w  0.0  0.0417  19.9711  17.8446  0.6322 -1.0907
                         JB  99.5%  0.5%  min   max   median
        sto1w -1.294762e+06   50.0   0.0  0.0  50.0  13.6258
        '''
        weekly_add["sto4w"] = (weekly_add["vol"].groupby("gvkey").rolling(4,min_periods=2).mean().droplevel(0)/\
                               weekly_add["shrout"]).clip(0,50)
        '''
        #########################
        ######### sto4w #########
        #########################

                  NA    ZERO    mean      std    skew    kurt
        sto4w  0.002  0.0337  21.033  17.7705  0.5565 -1.1802
                         JB  99.5%  0.5%  min   max   median
        sto4w -1.528160e+06   50.0   0.0  0.0  50.0  15.1294
        '''
        
        weekly_add["vtv1w"]		= np.log(weekly_add["stddolvol"]+1)
        '''
        str turnover of shares
        
        #########################
        ######### vtv1w #########
        #########################

                   NA    ZERO    mean     std   skew    kurt
        vtv1w  0.0012  0.0318  7.7209  2.5433 -0.543  1.3105
                        JB    99.5%  0.5%  min      max  median
        vtv1w -265102.2299  13.8409   0.0  0.0  18.4924  7.7423
        '''
        weekly_add["vtv4w"] = np.log(((
            weekly_add["dolvol2"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0)/\
            weekly_add["count1w"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0))-\
            (weekly_add["dolvol"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0)/\
             weekly_add["count1w"].groupby("gvkey").rolling(4,min_periods=2).sum().droplevel(0))**2)**.5).\
            replace([np.inf,-np.inf],np.nan)
        '''
        #########################
        ######### vtv4w #########
        #########################

                   NA  ZERO   mean     std    skew    kurt
        vtv4w  0.0079   0.0  8.602  2.2046  0.0676  0.8307
                         JB    99.5%    0.5%      min    max  median
        vtv4w -1.117570e+06  14.4665  3.2552 -14.2095  18.98  8.4987
        '''
        
        if False:
            weekly_add["mcap"]
        '''
        weekly market capitalisation

        #########################
        ########## mcap #########
        #########################

               NA  ZERO     mean     std    skew    kurt
        mcap  0.0   0.0  11.9758  2.2019  0.2277 -0.0731
                        JB    99.5%    0.5%     min      max   median
        mcap -1.488892e+06  17.9572  6.9583  2.0794  21.8031  11.8873
        '''
        
        ## market excess returns
        weekly_add["mer"]       = weekly_add["m1w"]*(1-weekly_add["b"].groupby("gvkey").shift(1))
        weekly_add["irv26w"]		= weekly_add["mer"].groupby("gvkey").rolling(26,min_periods=6).std().droplevel(0)
        '''
        weekly idiosyncratic return volatility
        
        #########################
        ######### irv26w ########
        #########################

                    NA  ZERO    mean     std     skew      kurt
        irv26w  0.0175   0.0  0.0629  0.1124  11.3191  427.2034
                          JB   99.5%    0.5%  min     max  median
        irv26w  4.866599e+08  0.6631  0.0011  0.0  10.429  0.0293
        '''

        weekly_add["irv156w"]		= weekly_add["mer"].groupby("gvkey").rolling(156,min_periods=27).std().droplevel(0)
        '''
        #########################
        ######## irv156w ########
        #########################

                    NA  ZERO   mean     std    skew      kurt
        irv156w  0.058   0.0  0.072  0.0996  6.5541  123.4274
                           JB   99.5%   0.5%  min     max  median
        irv156w  1.518417e+08  0.5748  0.002  0.0  4.9103  0.0389
        '''


        weekly_add["dso4w"]       = (weekly_add["shrout"]/weekly_add.groupby("gvkey").shift(4)["shrout"]-1).\
            replace([np.inf,-np.inf],np.nan).clip(-.99,10)
        '''
        4 week change in shares outstanding
        
        #########################
        ######### dso4w #########
        #########################

                   NA    ZERO   mean     std     skew       kurt
        dso4w  0.0078  0.7134  0.014  0.2061  31.1807  1299.8422
                         JB   99.5%    0.5%   min   max  median
        dso4w  2.694300e+09  0.7263 -0.1688 -0.99  10.0     0.0
        '''

        ####################
        #  Industry terms  #
        ####################
        
        weekly_add["SICCD"] = weekly_add["SICCD"].astype(str).str[:2]
        
        weekly_add_ind = pd.DataFrame()
        
        
        weekly_add["im1w"] = weekly_add["m1w"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["im1w"]    = weekly_add.groupby(["SICCD","year","week"])["im1w"].sum()
        del weekly_add["im1w"]
        
        weekly_add["im26w"] = weekly_add["m26w"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["im26w"]		= weekly_add.groupby(["SICCD","year","week"])["im26w"].sum()
        del weekly_add["im26w"]

        weekly_add["imcap"]         = np.exp(weekly_add["mcap"])
        weekly_add_ind["imcap"]		= weekly_add.groupby(["SICCD","year","week"])["imcap"].sum()
        del weekly_add["imcap"]
        
        weekly_add["irvar13w"] = weekly_add["rvar13w"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["irvar13w"]  = weekly_add.groupby(["SICCD","year","week"])["irvar13w"].sum()
        del weekly_add["irvar13w"]
        
        weekly_add["itv1w"] = weekly_add["tv1w"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["itv1w"]     = weekly_add.groupby(["SICCD","year","week"])["itv1w"].sum()
        del weekly_add["itv1w"]
        
        weekly_add["itv4w"] = weekly_add["tv4w"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["itv4w"]     = weekly_add.groupby(["SICCD","year","week"])["itv4w"].sum()
        del weekly_add["itv4w"]
        
        weekly_add["imdr4w"] = weekly_add["mdr4w"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["imdr4w"]    = weekly_add.groupby(["SICCD","year","week"])["imdr4w"].sum()
        del weekly_add["imdr4w"]
        
        weekly_add["ib"] = weekly_add["b"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["ib"]        = weekly_add.groupby(["SICCD","year","week"])["ib"].sum()
        del weekly_add["ib"]
        
        weekly_add["iirv26w"] = weekly_add["irv26w"]*np.exp(weekly_add["mcap"])
        weekly_add_ind["iirv26w"]    = weekly_add.groupby(["SICCD","year","week"])["iirv26w"].sum()
        del weekly_add["iirv26w"]
        
        
        weekly_add_ind["countgvkey"] = weekly_add.reset_index().groupby(["SICCD","year","week"])["gvkey"].count()
        
        if False:
            weekly_add[["ILL_absdolmov","relvol","ILL_numtrd"]]	= None
        '''
        1- (absolute dollar movement - 20% quantile)*
        1- (relative trading volume -20% quantile)*
        1- (numtrd -20% quantile)
        '''
        
        ## append new sub-dfs to initial list
        weekly.append(weekly_add)
        del weekly_add
        
        weekly_ind.append(weekly_add_ind)
        del weekly_add_ind
        
    ## concatenate subdfs
    weekly = pd.concat(weekly)
    weekly_ind = pd.concat(weekly_ind)
    weekly_ind = weekly_ind.groupby(["SICCD","year","week"]).sum()
    
    weekly_ind.reset_index(inplace=True)
    
    ## consolidate industries with too few observations
    count_ind = weekly_ind.groupby(["SICCD","year","week"])["countgvkey"].sum()
    count_ind = count_ind[count_ind<min_companies_per_industry_month]
    count_ind = count_ind.to_frame("NEWSIC")
    count_ind["NEWSIC"] = "NOIND"
    weekly_ind = pd.merge(weekly_ind,count_ind,left_on = ["SICCD","year","week"],
                            right_on = ["SICCD","year","week"],how="left")
    weekly_ind.loc[(~weekly_ind["NEWSIC"].isna())|\
                     (weekly_ind["SICCD"] == "Z"),"SICCD"] = "NOIND"
    del weekly_ind["NEWSIC"]
    
    weekly_ind = weekly_ind.groupby(["SICCD","year","week"]).sum()
    
    
    for ind_column in ["im1w","im26w","irvar13w","itv1w","itv4w","imdr4w","ib","iirv26w"]:
        weekly_ind[ind_column]        = weekly_ind[ind_column]/weekly_ind["imcap"]
        
    weekly_ind["imcap"]         = np.log(weekly_ind["imcap"]/weekly_ind["countgvkey"])
    
    
    
    weekly_ind                  = weekly_ind[[
        "im1w","im26w","imcap","irvar13w","itv1w","itv4w","imdr4w","ib","iirv26w"]]
    
    ## illiquidity ill
    quantile_border = .2
    weekly["ill"]       = 1
    ## absolute dollar movement
    weekly.reset_index(inplace=True)
    weekly_sample = pd.DataFrame()
    weekly_sample["ILL_absdolmov_sam"] = weekly.groupby(["year","week"])["ILL_absdolmov"].quantile(quantile_border)
    weekly_sample["sto_sam"] = weekly.groupby(["year","week"])["sto1w"].quantile(quantile_border)
    weekly_sample["ILL_numtrd_sam"] = weekly.groupby(["year","week"])["ILL_numtrd"].quantile(quantile_border)
    weekly = pd.merge(weekly,weekly_sample,left_on=["year","week"],right_on=["year","week"],
                      how = "left")
    del weekly_sample
    
    weekly["ill"]      = 1-(weekly["ILL_absdolmov"]/weekly["ILL_absdolmov_sam"]).clip(0,1)
    del weekly["ILL_absdolmov_sam"], weekly["ILL_absdolmov"]
    ## trading volume
    
    weekly["ill"]      *= (1-(weekly["sto1w"]/weekly["sto_sam"]).clip(0,1))
    del weekly["sto_sam"]
    ## weekly numtrd number of trades
    
    weekly.loc[~weekly["ILL_numtrd"].isna(),"ill"]      *= (1-(
        weekly.loc[~weekly["ILL_numtrd"].isna(),"ILL_numtrd_sam"]-\
        weekly.loc[~weekly["ILL_numtrd"].isna(),"ILL_numtrd"]).clip(0,1))
    weekly.loc[~weekly["ILL_numtrd"].isna(),"ill"] = \
        weekly.loc[~weekly["ILL_numtrd"].isna(),"ill"]**(1/3)
    weekly.loc[weekly["ILL_numtrd"].isna(),"ill"] = \
        weekly.loc[~weekly["ILL_numtrd"].isna(),"ill"]**(1/2)
    del weekly["ILL_numtrd_sam"], weekly["ILL_numtrd"]
    '''
    #########################
    ########## ill ##########
    #########################

             NA    ZERO    mean     std    skew     kurt
    ill  0.0329  0.9093  0.0351  0.1518  4.6291  21.3571
                   JB   99.5%  0.5%  min  max  median
    ill  5.407158e+07  0.9942   0.0  0.0  1.0     0.0
    '''
    
    del count_ind
    weekly = pd.merge(weekly, weekly_ind, left_on = ["SICCD","year","week"],
                      right_on = ["SICCD","year","week"], how = "left")
    
    
    weekly["im1w"] -= weekly["m1w"]
    '''
    #########################
    ########## im1w #########
    #########################

              NA  ZERO    mean     std    skew     kurt
    im1w  0.0117   0.0  0.0018  0.0833 -7.5279  438.592
                    JB  99.5%    0.5%     min     max  median
    im1w  3.440850e+08  0.256 -0.3362 -9.9901  1.4176  0.0039
    '''
    
    weekly["im26w"] -= weekly["m26w"]
    '''
    #########################
    ######### im26w #########
    #########################

               NA  ZERO    mean     std     skew      kurt
    im26w  0.0367   0.0  0.0244  0.2948 -10.4513  708.9787
                     JB  99.5%    0.5%      min     max  median
    im26w  5.937948e+08  0.714 -1.2219 -53.1957  2.2426  0.0351
    '''

    
    weekly["irvar13w"]  -=weekly["rvar13w"]
    '''
    #########################
    ######## irvar13w #######
    #########################

                  NA  ZERO   mean     std     skew      kurt
    irvar13w  0.0194   0.0 -0.001  0.0045 -27.8066  1958.428
                        JB   99.5%    0.5%     min     max  median
    irvar13w  2.622843e+09  0.0024 -0.0221 -1.1236  0.0413 -0.0001
    '''
    
    weekly["itv1w"]     -=weekly["tv1w"]
    '''
    #########################
    ######### itv1w #########
    #########################

               NA  ZERO    mean     std   skew    kurt
    itv1w  0.0117   0.0  3.5571  2.1726  0.066 -0.2459
                     JB   99.5%    0.5%     min      max  median
    itv1w -1.677331e+06  9.1273 -1.4079 -3.1561  14.5586  3.5628
    '''
    
    weekly["itv4w"]     -=weekly["tv4w"]
    '''
    #########################
    ######### itv4w #########
    #########################

               NA  ZERO    mean     std    skew    kurt
    itv4w  0.0136   0.0  3.5358  2.1733  0.0532 -0.2302
                     JB   99.5%    0.5%      min      max  median
    itv4w -1.672385e+06  9.0867 -1.4542 -14.5684  15.0673  3.5447
    '''
    
    weekly["imdr4w"]    -=weekly["mdr4w"]
    '''
    #########################
    ######### imdr4w ########
    #########################

                NA  ZERO    mean     std   skew    kurt
    imdr4w  0.0175   0.0 -0.0191  0.0669 -5.222  64.361
                      JB   99.5%   0.5%     min     max  median
    imdr4w  8.855090e+07  0.0751 -0.363 -2.3709  0.4946 -0.0028
    '''
    
    weekly["ib"]        -=weekly["b"]
    '''
    #########################
    ########### ib ##########
    #########################

            NA  ZERO    mean     std   skew    kurt
    ib  0.0175   0.0  0.1886  1.0258 -0.025  4.0533
                 JB   99.5%    0.5%     min     max  median
    ib  548520.7189  3.8295 -3.5865 -5.9216  7.4022  0.2245
    '''
    
    weekly["iirv26w"]   -=weekly["irv26w"]
    '''
    #########################
    ######## iirv26w ########
    #########################

                NA  ZERO    mean     std     skew      kurt
    iirv26w  0.029   0.0 -0.0314  0.1114 -11.5082  444.9963
                       JB   99.5%    0.5%      min     max  median
    iirv26w  5.048763e+08  0.0916 -0.6231 -10.3971  0.2035 -0.0025
    '''
    
    weekly["imcap"] -= weekly["mcap"]
    '''
    #########################
    ######### imcap #########
    #########################

               NA  ZERO    mean     std    skew    kurt
    imcap  0.0117   0.0  1.8755  2.0008  0.0325 -0.1295
                     JB   99.5%    0.5%     min      max  median
    imcap -1.623768e+06  6.9776 -3.1039 -5.3934  12.7863   1.867
    '''
    
    weekly_date = weekly.groupby(["year","week"])["date"].max()
    del weekly["date"]
    weekly = pd.merge(weekly,weekly_date,left_on = ["year","week"], right_on = ["year","week"],how="left")
    
    ## earnings announcement dates incorporation
    ea = pd.read_hdf(PATH_DATA+"earnings_announcement_raw.h5","data")
    ea["date"] = ea.index.get_level_values("date")
    ea["week"] = ea["date"].dt.isocalendar().week
    ea["year"] = ea["date"].dt.isocalendar().year
    del ea["date"]
    weekly = pd.merge(weekly,ea,left_on = ["gvkey","year","week"],
                      right_on = ["gvkey","year","week"], how = "left")
    del ea
    '''
    #########################
    ########## EAR ##########
    #########################

             NA  ZERO    mean     std    skew     kurt
    EAR  0.9482   0.0  0.0003  0.0344  2.1521  72.6913
                   JB   99.5%    0.5%     min     max  median
    EAR  4.583384e+07  0.1264 -0.1143 -0.8976  1.8617 -0.0003
    
    #########################
    ########## EAV ##########
    #########################

             NA    ZERO    mean     std    skew     kurt
    EAV  0.9485  0.0003  1.6126  1.5209  2.9411  11.3535
                   JB  99.5%  0.5%  min   max  median
    EAV  2.231646e+07   10.0   0.0  0.0  10.0   1.219
    '''
    
    columns_finished_weekly = [
        "b", "b2", "bas", "dso4w", "ill", "im1w", "im26w", "imcap", "irv26w", "irv156w", 
        "mcap", "mdr1w", "mdr4w", "sto1w", "tv1w", "vst1w", "vtv1w", "sto4w", "tv4w", "vst4w", 
        "vtv4w", "ztd1w", "m1w", "m4w", "m13w", "m26w", "ret1w", "ret4w", "ret13w", 
        "rvar1w", "rvar4w", "rvar13w", "rvar26w", "rvarf1w", "rvarf4w", "rvarf13w", "EAR", 
        "EAV", "irvar13w", "itv1w", "itv4w", "imdr4w", "ib", "iirv26w","rq1w", "rq4w", "rq13w"]
    
    weekly.drop_duplicates(subset=["gvkey","date"],inplace=True)
    weekly["date"] = pd.to_datetime(weekly["date"]).apply(
        lambda x: x.date())#.astype('datetime64[ns]')
    weekly.set_index(["gvkey","date"],inplace=True)
    
    # weekly.drop_duplicates(inplace=True)
    
    weekly = weekly[~weekly.index.duplicated(keep="first")]
    
    for column in ["rq1w","rq4w","rq13w"]:
        weekly[column] = weekly[column].clip(0,.5)
    for column in ["rvar1w","rvar4w","rvar13w","rvar26w","rvarf1w","rvarf4w","rvarf13w"]:
        weekly[column] = weekly[column].clip(0,.1)
    
    weekly[columns_finished_weekly].to_hdf(PATH_DATA+"weekly_P3.h5",key="data")
    
####################################################################################################
####################################################################################################
#######################################  Monthly (from CRSP)  ######################################
####################################################################################################
####################################################################################################

if ("FULL" in mode or "monthly" in mode):
    ###########################
    #  Monthly Preprocessing  #
    ###########################
    
    #### import and prepare data
    
    ## load matching table CRSP_Compustat
    matching_table = pd.read_hdf(PATH_DATA+"CRSP_Compustat_match.h5","data")
    
    ## quarterly signals needed for monthly calculations
    quarterly_base = pd.read_hdf(PATH_DATA+"quarterly_to_monthly.h5","data")
    quarterly_base.reset_index(inplace=True)
    quarterly_base["date"] = quarterly_base["date"].astype(int).astype(str).apply(
        lambda x: dt.datetime.strptime(x,"%Y%m%d"))
    quarterly_base.set_index(["gvkey","date"],inplace=True)
    quarterly_cols = quarterly_base.columns
    
    ## loading monthly CRSP data
    monthly_CRSP = pd.read_csv(PATH_DATA+"monthly.txt",sep="\t")
    
    ## merge with gvkey
    monthly_CRSP = pd.merge(matching_table[["gvkey","CUSIP"]],monthly_CRSP,
                     left_on ="CUSIP",right_on = "CUSIP",how="inner")
    del matching_table
    
    #### preprocessing the relevant columns
    del monthly_CRSP["CUSIP"]
    monthly_CRSP.dropna(subset= ["PRC","SHROUT"],inplace=True)
    monthly_CRSP.drop_duplicates(subset=["gvkey","date"],keep="first")
    for column in ["RET","DLRET","PRC","SHROUT","VOL"]:
        monthly_CRSP[column] = pd.to_numeric(
            monthly_CRSP[column],errors= "coerce").fillna(0)
    monthly_CRSP['date'] = pd.to_datetime(monthly_CRSP['date'], format="%Y%m%d") + MonthEnd(0)
    monthly_CRSP.set_index(["gvkey","date"],inplace=True)
    monthly_CRSP.loc[monthly_CRSP["RET"].isin([-66,-77,-88,-99]),"RET"] = 0
    # monthly_CRSP["DLRET"] = pd.to_numeric(daily["DLRET"],errors= "coerce").fillna(0)
    monthly_CRSP.loc[monthly_CRSP["DLRET"].isin([-66,-77,-88,-99]),"DLRET"] = 0
    monthly_CRSP["RET"] = ((1+monthly_CRSP["RET"])*(1+monthly_CRSP["DLRET"])-1).clip(-.99,10)
    
    ## merge with quarterly table
    monthly_CRSP["monthly_datapoint"] = 1
    monthly_CRSP = pd.merge(monthly_CRSP, quarterly_base, left_on = ["gvkey","date"],
                            right_on = ["gvkey","date"],how="outer")
    
    ## drop duplicates based on gvkey date duplicates and the NA values in that row.
    monthly_CRSP["NA_Row"] = monthly_CRSP.isna().sum(axis=1)
    monthly_CRSP.sort_values("NA_Row",inplace=True,ascending=True)
    monthly_CRSP.reset_index(inplace=True)
    monthly_CRSP.drop_duplicates(subset=["gvkey","date"],inplace=True)
    monthly_CRSP.set_index(["gvkey","date"],inplace=True)
    del monthly_CRSP["NA_Row"]
    
    del quarterly_base
    ## forward quarterly report data up to 4 months
    monthly_CRSP.sort_index(inplace=True)
    for column in quarterly_cols:
        monthly_CRSP[column] = monthly_CRSP[column].groupby("gvkey").ffill(4)
    ## delete none monthly datapoints
    monthly_CRSP = monthly_CRSP[monthly_CRSP["monthly_datapoint"]==1]
        
    ###################################
    #  Monthly predictor calculation  #
    ###################################
    
    monthly_CRSP["mcap"] = monthly_CRSP["SHROUT"]*monthly_CRSP["PRC"].abs()
    
    ## quarterly columns to be preserved
    monthly_CRSP["bmq"] = monthly_CRSP["BM"].copy()
    monthly_CRSP["cfpq"] = monthly_CRSP["CFP"].copy()
    monthly_CRSP["epq"] =  monthly_CRSP["EP"].copy()
    monthly_CRSP["spq"] =  monthly_CRSP["SP"].copy()
    monthly_CRSP["rdmcapq"] =  monthly_CRSP["RDMCAP"].copy()
    '''
    #########################
    ########## mcap #########
    #########################

           NA  ZERO          mean           std     skew       kurt
    mcap  0.0   0.0  2.109948e+06  1.681519e+07  56.5669  6083.4506
                    JB         99.5%      0.5%     min           max    median
    mcap  2.977721e+09  6.388284e+07  970.1547  3.7188  2.902368e+09  122076.0
    '''
    
    ## book to market
    monthly_CRSP["BM"] = (1000*monthly_CRSP["bmq"]/monthly_CRSP["mcap"]).clip(-1.5,50)
    '''
    #########################
    ########### BM ##########
    #########################

           NA  ZERO    mean     std    skew      kurt
    BM  0.274   0.0  1.2063  4.0609  9.9149  107.5942
                  JB    99.5%  0.5%  min   max  median
    BM  7.851371e+07  41.3621  -1.5 -1.5  50.0  0.5898
    '''
    
    # cash flow to price
    monthly_CRSP["CFP"] = (monthly_CRSP["cfpq"]/monthly_CRSP["mcap"]).clip(-1,2)
    '''
    #########################
    ########## CFP ##########
    #########################

             NA    ZERO    mean     std     skew       kurt
    CFP  0.0843  0.1439  0.0002  0.0132  59.9017  9073.9964
                   JB   99.5%    0.5%  min  max  median
    CFP  3.694422e+09  0.0107 -0.0054 -1.0  2.0     0.0
    '''
    
    # dividend to price ratio
    monthly_CRSP.loc[:,"DIVP"] = 4*monthly_CRSP["DIVAMT"]/monthly_CRSP["PRC"].abs()
    monthly_CRSP.loc[:,"DIVP"] = monthly_CRSP["DIVP"].groupby("gvkey").ffill(4)
    '''
    #########################
    ########## DIVP #########
    #########################

              NA    ZERO    mean     std      skew        kurt
    DIVP  0.4981  0.0254  0.0505  0.5033  106.7191  17143.8163
                    JB   99.5%  0.5%     min    max  median
    DIVP  9.888563e+09  0.4587   0.0 -0.2912  124.8  0.0267
    '''
    
    ## momentum factors
    monthly_CRSP.loc[:,"M1M"] = np.log(monthly_CRSP["RET"]+1)
    monthly_CRSP.loc[:,"M6M"] = monthly_CRSP["M1M"].groupby("gvkey").rolling(6,2).sum().droplevel(0)-\
        monthly_CRSP["M1M"]
    monthly_CRSP.loc[:,"M12M"] = monthly_CRSP["M1M"].groupby("gvkey").rolling(12,7).sum().droplevel(0)-\
        monthly_CRSP["M1M"] - monthly_CRSP["M6M"]
    monthly_CRSP.loc[:,"M36M"] = monthly_CRSP["M1M"].groupby("gvkey").rolling(36,13).sum().droplevel(0)-\
        monthly_CRSP["M1M"] - monthly_CRSP["M6M"] - monthly_CRSP["M12M"]
    
    monthly_CRSP.loc[:,"DM6M"] = monthly_CRSP["M6M"] - monthly_CRSP["M6M"].groupby("gvkey").shift(1)
    
    for column in ["M1M","M6M","M12M","M36M","DM6M"]:
        monthly_CRSP.loc[:,column] = monthly_CRSP.loc[:,column].clip(-5,5)
    #     monthly_CRSP.loc[:,column] = np.exp(monthly_CRSP[column])-1
    '''
    #########################
    ########## M1M ##########
    #########################

          NA    ZERO    mean     std   skew     kurt
    M1M  0.0  0.0412 -0.0015  0.1535 -0.749  20.3232
                   JB   99.5%    0.5%     min     max  median
    M1M  3.086145e+06  0.5334 -0.5896 -4.6052  2.3979     0.0
    
    #########################
    ########## M6M ##########
    #########################

             NA    ZERO    mean     std    skew    kurt
    M6M  0.0074  0.0083 -0.0045  0.3379 -0.8906  7.7283
                   JB   99.5%    0.5%  min     max  median
    M6M  1.246127e+06  1.0343 -1.3646 -5.0  4.2187  0.0185
    
    #########################
    ########## M12M #########
    #########################

              NA    ZERO    mean     std   skew    kurt
    M12M  0.0444  0.0078  0.0002  0.3598 -0.748  6.5961
                  JB   99.5%    0.5%  min     max  median
    M12M  920154.695  1.1148 -1.4109 -5.0  4.2182   0.024
    
    #########################
    ########## M36M #########
    #########################

              NA    ZERO    mean    std    skew    kurt
    M36M  0.0878  0.0074  0.0364  0.679 -0.9699  4.4767
                   JB   99.5%   0.5%  min  max  median
    M36M  826370.2028  1.8781 -2.623 -5.0  5.0  0.0962
    
    #########################
    ########## DM6M #########
    #########################

              NA    ZERO    mean     std    skew    kurt
    DM6M  0.0149  0.0058 -0.0005  0.2088  0.0513  9.7466
                    JB   99.5%    0.5%     min     max  median
    DM6M  1.065732e+06  0.7623 -0.7487 -4.6052  4.5957    -0.0
    '''
    
    monthly_CRSP.loc[:,"EP"] = (1000*monthly_CRSP["epq"]/monthly_CRSP["mcap"]).clip(-10,10)
    
    '''
    #########################
    ########### EP ##########
    #########################

            NA    ZERO    mean    std    skew     kurt
    EP  0.2306  0.0021 -0.0213  1.157 -0.0049  52.3443
                  JB   99.5%   0.5%   min   max  median
    EP  7.782628e+06  7.6754 -7.371 -10.0  10.0  0.0448
    '''
    
    monthly_CRSP.loc[:,"MCAP"] = np.log(monthly_CRSP["mcap"])
    '''
    #########################
    ########## MCAP #########
    #########################

           NA  ZERO     mean     std    skew    kurt
    MCAP  0.0   0.0  11.8279  2.2438  0.2734 -0.1054
                   JB    99.5%    0.5%     min      max   median
    MCAP -442610.3864  17.9726  6.8775  1.3134  21.7888  11.7124
    '''
    
    monthly_CRSP.loc[:,"SP"] = (1000*monthly_CRSP["spq"]/monthly_CRSP["mcap"]).clip(-5,100)
    '''
    #########################
    ########### SP ##########
    #########################

            NA   ZERO    mean      std    skew     kurt
    SP  0.2286  0.014  3.3115  10.5165  7.3837  60.1743
                  JB  99.5%  0.5%  min    max  median
    SP  4.341155e+07  100.0   0.0 -5.0  100.0  0.9417
    '''
    
    monthly_CRSP["DEP1M"]       = monthly_CRSP["EP"]-monthly_CRSP["EP"].groupby("gvkey").shift(1)
    '''
    #########################
    ######### DEP1M #########
    #########################

               NA    ZERO   mean     std    skew      kurt
    DEP1M  0.2403  0.0256 -0.007  0.3557 -1.8838  766.4042
                     JB   99.5%    0.5%   min   max  median
    DEP1M  1.226398e+08  0.8724 -1.1681 -20.0  20.0     0.0
    '''
    
    monthly_CRSP["DEP12M"]      = monthly_CRSP["EP"]-monthly_CRSP["EP"].groupby("gvkey").shift(12)
    '''
    #########################
    ######### DEP12M ########
    #########################

                NA    ZERO    mean     std    skew      kurt
    DEP12M  0.3045  0.0029 -0.0491  1.0726 -1.4836  106.9675
                      JB   99.5%    0.5%   min   max  median
    DEP12M  1.778583e+07  3.9641 -6.3076 -20.0  20.0 -0.0008
    '''
    
    monthly_CRSP["DCFP1M"]      = monthly_CRSP["CFP"]-monthly_CRSP["CFP"].groupby("gvkey").shift(1)
    '''
    #########################
    ######### DCFP1M ########
    #########################

                NA    ZERO  mean     std     skew        kurt
    DCFP1M  0.0957  0.1607  -0.0  0.0057 -50.8142  36665.5091
                      JB   99.5%    0.5%     min    max  median
    DCFP1M  7.411205e+09  0.0018 -0.0021 -2.0701  1.791     0.0
    '''
    
    monthly_CRSP["DCFP12M"]     = monthly_CRSP["CFP"]-monthly_CRSP["CFP"].groupby("gvkey").shift(12)
    '''
    #########################
    ######## DCFP12M ########
    #########################

                 NA    ZERO    mean     std     skew       kurt
    DCFP12M  0.1742  0.1235 -0.0001  0.0151 -12.9905  9042.3095
                       JB   99.5%    0.5%  min  max  median
    DCFP12M  1.532102e+09  0.0075 -0.0093 -3.0  3.0     0.0
    '''
    
    monthly_CRSP["DSP1M"]       = monthly_CRSP["SP"]-monthly_CRSP["SP"].groupby("gvkey").shift(1)
    '''
    #########################
    ######### DSP1M #########
    #########################

               NA    ZERO    mean     std    skew      kurt
    DSP1M  0.2382  0.0374  0.0279  1.7331  3.1697  701.2566
                     JB   99.5%    0.5%    min    max  median
    DSP1M  1.164644e+08  5.4756 -4.6981 -105.0  100.0     0.0
    '''
    
    monthly_CRSP["DSP12M"]      = monthly_CRSP["SP"]-monthly_CRSP["SP"].groupby("gvkey").shift(12)
    '''
    #########################
    ######### DSP12M ########
    #########################

                NA    ZERO    mean     std    skew      kurt
    DSP12M  0.3024  0.0124  0.1962  4.8353  2.6663  140.2783
                      JB    99.5%     0.5%    min    max  median
    DSP12M  2.613593e+07  22.1754 -16.2144 -105.0  105.0  0.0067
    '''
    
    ############################
    #  Monthly Industry Table  #
    ############################
    
    monthly_CRSP["SICCD"] = monthly_CRSP["SICCD"].groupby("gvkey").ffill()
    monthly_CRSP["SICCD"] = monthly_CRSP["SICCD"].groupby("gvkey").bfill()
    monthly_CRSP["SIC"] = monthly_CRSP["SICCD"].astype(str)
    monthly_CRSP["SIC"] = monthly_CRSP["SIC"].str[:2]
    
    count_ind = monthly_CRSP.groupby(["SIC","date"])["SIC"].count()
    count_ind = count_ind[count_ind<min_companies_per_industry_month]
    count_ind = count_ind.to_frame("NEWSIC")
    count_ind["NEWSIC"] = "NOIND"
    monthly_CRSP.reset_index(inplace=True)
    monthly_CRSP = pd.merge(monthly_CRSP,count_ind,left_on = ["SIC","date"],
                            right_on = ["SIC","date"],how="left")
    del count_ind
    monthly_CRSP.loc[(~monthly_CRSP["NEWSIC"].isna())|\
                     (monthly_CRSP["SIC"] == "Z"),"SIC"] = "NOIND"
    del monthly_CRSP["NEWSIC"]
    
    monthly_industry = pd.DataFrame()
    
    monthly_industry["IBM"] = (1000*monthly_CRSP.groupby(["SIC","date"])["bmq"].sum()/\
        monthly_CRSP.groupby(["SIC","date"])["mcap"].sum()).clip(-1.5,50)

    monthly_industry["ICFP"] = (1000*monthly_CRSP.groupby(["SIC","date"])["cfpq"].sum()/\
        monthly_CRSP.groupby(["SIC","date"])["mcap"].sum()).clip(-1,2)
        
    monthly_CRSP["im6m"] = (np.exp(monthly_CRSP["M6M"])-1)*monthly_CRSP["mcap"]
    monthly_industry["IM6M"] = np.log(monthly_CRSP.groupby(["SIC","date"])["im6m"].sum()/\
         monthly_CRSP.groupby(["SIC","date"])["mcap"].sum()+1).clip(-10,10)  
    del monthly_CRSP["im6m"]
    
    monthly_CRSP["im12m"] = (np.exp(monthly_CRSP["M12M"])-1)*monthly_CRSP["mcap"]
    monthly_industry["IM12M"] = np.log(monthly_CRSP.groupby(["SIC","date"])["im12m"].sum()/\
         monthly_CRSP.groupby(["SIC","date"])["mcap"].sum()+1).clip(-10,10)  
    del monthly_CRSP["im12m"]
    
    monthly_industry["IMCAP"] = monthly_CRSP.groupby(["SIC","date"])["MCAP"].sum()/\
        monthly_CRSP.groupby(["SIC","date"])["mcap"].count()

    monthly_industry["IEP"] = (1000*monthly_CRSP.groupby(["SIC","date"])["epq"].sum()/\
        monthly_CRSP.groupby(["SIC","date"])["mcap"].sum()).clip(-10,10)
        
    monthly_industry["ISP"] = (1000*monthly_CRSP.groupby(["SIC","date"])["spq"].sum()/\
        monthly_CRSP.groupby(["SIC","date"])["mcap"].sum()).clip(-5,100)
        
    monthly_CRSP = pd.merge(monthly_CRSP,monthly_industry,left_on = ["SIC","date"],
                            right_on = ["SIC","date"],how="left")
    monthly_CRSP.set_index(["gvkey","date"],inplace=True)

    monthly_CRSP["IBM"] = monthly_CRSP["BM"]- monthly_CRSP["IBM"]
    '''
    #########################
    ########## IBM ##########
    #########################

             NA  ZERO    mean    std     skew      kurt
    IBM  0.6223   0.0  0.5882  3.541  11.4057  143.5351
                   JB   99.5%    0.5%      min      max  median
    IBM  1.042333e+08  28.623 -1.8159 -19.1794  49.9692  0.1142
    '''
    
    monthly_CRSP["ICFP"] = monthly_CRSP["CFP"]- monthly_CRSP["ICFP"]
    '''
    #########################
    ########## ICFP #########
    #########################

           NA    ZERO    mean     std    skew     kurt
    ICFP  0.0  0.0255  0.1051  0.1725  3.1189  42.2676
                   JB   99.5%    0.5%  min  max  median
    ICFP  133239.1543  0.9592 -0.5644 -1.0  2.0  0.0846
    '''
    
    monthly_CRSP["IM6M"] = monthly_CRSP["M6M"]- monthly_CRSP["IM6M"]
    '''
    #########################
    ########## IM6M #########
    #########################

           NA    ZERO    mean     std    skew    kurt
    IM6M  0.0  0.0006  0.0642  0.1656 -0.0804  3.3089
                JB   99.5%    0.5%     min     max  median
    IM6M  570.4315  0.5705 -0.4655 -1.0988  2.1299  0.0679
    '''
    
    monthly_CRSP["IM12M"] = monthly_CRSP["M12M"]- monthly_CRSP["IM12M"]
    '''
    #########################
    ######### IM12M #########
    #########################

            NA   ZERO    mean     std    skew    kurt
    IM12M  0.0  0.004  0.0763  0.1805 -0.0781  3.9641
                  JB   99.5%    0.5%     min     max  median
    IM12M  1684.7023  0.6341 -0.5008 -1.2159  2.7154  0.0783
    '''
    
    monthly_CRSP["IMCAP"] = monthly_CRSP["MCAP"]- monthly_CRSP["IMCAP"]
    '''
    #########################
    ######### IMCAP #########
    #########################

            NA  ZERO     mean     std    skew    kurt
    IMCAP  0.0   0.0  11.8743  1.5582  0.3053 -0.7909
                 JB    99.5%    0.5%     min      max   median
    IMCAP -5825.659  15.4977  8.9067  8.1784  16.3622  11.6483
    '''
    
    monthly_CRSP["IEP"] = monthly_CRSP["EP"]- monthly_CRSP["IEP"]
    '''
    #########################
    ########## IEP ##########
    #########################

          NA    ZERO    mean     std     skew      kurt
    IEP  0.0  0.0259  0.0625  0.1224 -16.8561  778.3992
                   JB   99.5%    0.5%     min     max  median
    IEP  3.258529e+06  0.4825 -0.2516 -6.2606  1.6822  0.0556
    '''
    
    monthly_CRSP["ISP"] = monthly_CRSP["SP"]- monthly_CRSP["ISP"]
    '''
    #########################
    ########## ISP ##########
    #########################

          NA    ZERO    mean     std    skew    kurt
    ISP  0.0  0.0262  1.7395  1.9449  4.8161  38.525
                  JB    99.5%  0.5%  min      max  median
    ISP  218674.2479  13.0349   0.0  0.0  37.0774  1.2189
    '''
    
    #### RVAR based on daily data imported from previously calculated.
    monthly_rvar = pd.read_hdf(PATH_DATA+"monthly_rvar.h5","data")
    
    monthly_CRSP = pd.merge(monthly_CRSP,monthly_rvar, left_index=True,
                            right_index=True, how = "left")
    del monthly_rvar
    monthly_CRSP["RVAR6M"] = monthly_CRSP["RVAR6M"].clip(0,.15)
    monthly_CRSP["RVAR1M"] = monthly_CRSP["RVAR1M"].clip(0,.15)
    
    columns_monthly = ["BM", "CFP", "DIVP", "DM6M", "EP", "IBM", "ICFP", "IM6M", 
                       "IM12M", "IMCAP", "M1M", "M6M", "M12M", "M36M", "MCAP", "SP", 
                       "RVAR1M", "RVAR6M", "DEP1M", "DEP12M", "IEP", "DCFP1M", "DCFP12M", 
                       "DSP1M", "DSP12M", "ISP"]
    
    monthly_CRSP["NA_Row"] = monthly_CRSP.isna().sum(axis=1)
    monthly_CRSP.sort_values("NA_Row",inplace=True,ascending=True)
    monthly_CRSP.reset_index(inplace=True)
    monthly_CRSP.drop_duplicates(subset=["gvkey","date"],inplace=True)
    monthly_CRSP["date"] = pd.to_datetime(monthly_CRSP["date"]).apply(
        lambda x: x.date())#.astype('datetime64[ns]')
    monthly_CRSP.set_index(["gvkey","date"],inplace=True)
    del monthly_CRSP["NA_Row"]
    
    monthly_CRSP[columns_monthly].to_hdf(PATH_DATA+"monthly_P3.h5",key="data")
    del monthly_CRSP

    
####################################################################################################
####################################################################################################
################################  Quarterly (from Compustat) [2/2]  ################################
####################################################################################################
####################################################################################################

@nb.jit(parallel=True)
def is_in_set_pnb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
            continue
    return result.reshape(shape)

if ("FULL" in mode or "quarterly" in mode or "ind" in mode):  
    
    ##################################################
    ##################################################
    ########  Quarterly: Industry Adjustments  #######
    ##################################################
    ##################################################
    
    '''
    Industry demeaning measures.
    
    IDCE        : Industry adjusted % change in capital expenditures 
    IDE         : Industry adjusted employee growth
    ISCH        : Industry sales concentration
    
    
    Problem1: rolling windows on a quarterly base. 
    Problem2: Market Capitalisation values.
    
    So far: We used end of the month data, so we could just use end of the month market caps. 
    Now, we have the problem that the dates are not only more continuous, but also the market 
    capitalisation might be biased, when we use market cap values too far back.
    
    Option 1. Ignore part of this issue. Just use the quarterly market cap values. How much can they change?
        How biased can the measures be?
    Option 2. Extract all dates, extract market cap for these dates for all companies.
        How many dates in the financial calendar are report dates?
        Answer: almost all of them.
    
    '''
    quarterly = pd.read_hdf(PATH_DATA+"quarterly_P3_s2.h5",key = "data")
    quarterly = pd.merge(
        pd.read_hdf(PATH_DATA+"quarterly_to_monthly.h5","data")["SP"], quarterly,
        left_index=True,right_on = ["gvkey","date"],how="inner")
    quarterly_industry = pd.read_csv(PATH_DATA+"quarterly.csv", delimiter = ",",usecols=["gvkey","sic"])
    quarterly_industry = quarterly_industry.dropna(subset="sic").drop_duplicates()
    quarterly.reset_index(inplace=True)
    quarterly = pd.merge(quarterly, quarterly_industry, left_on = ["gvkey"],right_on = ["gvkey"],
                         how="left")
    quarterly["date"] = quarterly["date"].astype(int).apply(lambda x: dt.datetime.strptime(str(x),"%Y%m%d")).dt.date
    quarterly.set_index(["gvkey","date"],inplace=True)
    
    quarterly = quarterly[~quarterly.index.duplicated(keep="first")]
    
    #### import mcap figures
    daily_MCAP = pd.read_hdf(PATH_DATA+"daily_MCAP.h5","data").reset_index()
    daily_MCAP["date"] =  pd.to_datetime(daily_MCAP["date"],errors="coerce").dt.date
    daily_MCAP = daily_MCAP[daily_MCAP["date"].isin(
        quarterly.index.get_level_values("date").unique())]
    
    ## extract unique industries
    quarterly["sic"] = quarterly["sic"].apply(lambda x: "{:04}".format(x))
    quarterly["sic"] = quarterly["sic"].str[:2]
    industries = quarterly["sic"].unique().tolist()
    industries.sort()
    
    company_ind = quarterly["sic"].reset_index()[["gvkey","sic"]].drop_duplicates()
    daily_MCAP = pd.merge(daily_MCAP,company_ind,left_on = ["gvkey"],right_on=["gvkey"],
                          how="inner")
    del company_ind
    
    #### cut out columns needed:
    ## columns needed:
    quarterly_columns_needed = [
        "IDAT","DE","IDPM","ACA","SP","ROI","ROE","OP","RDS","DS","FSS","DEAR","DCE","sic"]
    quarterly = quarterly[quarterly_columns_needed]
    
    
    columns_ind_q = ["IDCE", "IDE", "IDPM", "ISCH", "ISS", "IACA", "IROI", "IROE", "IOP", "IRDS", 
               "IDS", "IFSS", "IDEAR","IDAT"]
    
    #### int_date
    quarterly.reset_index(inplace=True); 
    quarterly["int_date"] = quarterly["date"].apply(lambda x: (x-dt.date(1960,1,1)).days)
    daily_MCAP["int_date"] = daily_MCAP["date"].apply(lambda x: (x-dt.date(1960,1,1)).days)
    
    
    ## groupby datasets
    quarterly_group = {x:y for x,y in quarterly.groupby("sic")}
    daily_MCAP_group = {x:y for x,y in daily_MCAP.groupby("sic")}
    ind_data = []
    skipcounter= 0
    
    industries = quarterly["sic"].unique().tolist()
    industries.sort()
    columns_local_q = {j:i for i, j in enumerate(quarterly_group[industries[0]].columns)}
    cutout_columns = [columns_local_q[i] for i in columns_local_q.keys() if i in ["gvkey","date"]]
    columns_local_q["MCAP"] = len(columns_local_q)
    columns_local_d = {j:i for i, j in enumerate(daily_MCAP_group[industries[0]].columns)}
    
    subdata_q = {}
    subdata_d = {}
    
    ## for reasons of performance enhancement, we split up the data into five year intervals.
    for industry in industries:
        int_date = quarterly["int_date"].min()
        while quarterly_group[industry].shape[0]>0:
            ## date ranges
            old_int_date = int(int_date)
            int_date+=5*365.2524
            min_date=int_date -100
            
            ## cut out data quarterly
            subdata_q[f"{industry}_{old_int_date:d}"] = quarterly_group[industry][
                quarterly_group[industry]["int_date"]<=int_date]
            quarterly_group[industry] = quarterly_group[industry][
                quarterly_group[industry]["int_date"]>min_date]
            
            ## cut out data daily
            subdata_d[f"{industry}_{old_int_date:d}"] = daily_MCAP_group[industry][
                daily_MCAP_group[industry]["int_date"]<=int_date]
            daily_MCAP_group[industry] = daily_MCAP_group[industry][
                daily_MCAP_group[industry]["int_date"]>min_date]
            
        del quarterly_group[industry]
        del daily_MCAP_group[industry]
    
    subdata_q["noind"] = []
    subdata_d["noind"] = []
    industries = list(subdata_q.keys())
    
    total_bar = quarterly[['int_date', 'sic']].drop_duplicates().shape[0]
    
    progress_bar = ELK_progress_bar(
        n=[len(industries),1], bar_width=[25,25], sign=[u"■",u"■"],
        name=["Industries","side_bar"])
    
    # with alive_bar(total_bar) as bar_ind:
    for industry in industries:
        ########################################
        #####  Standard Industry Variables  ####
        ########################################
        # industry = industries[2]
        # industry = "noind"
        # print("Quarterly ind:",industry)
        
        if industry != "noind":
            
            quarterly_local = subdata_q[industry].sort_values(["gvkey","date"]).to_numpy()
            del subdata_q[industry]
            
            daily_local = subdata_d[industry].sort_values(["gvkey","date"],ascending=False).to_numpy()
            del subdata_d[industry]
            # a_time, b_time = 0, []
        else:
            subdata_q["noind"] = np.concatenate(subdata_q["noind"])
            subdata_q["noind"] = pd.DataFrame(subdata_q["noind"]).drop_duplicates()
            quarterly_local = subdata_q["noind"].sort_values([0,1]).to_numpy()
            del subdata_q[industry]
            
            subdata_d["noind"] = np.concatenate(subdata_d["noind"])
            subdata_d["noind"] = pd.DataFrame(subdata_d["noind"]).drop_duplicates()
            daily_local = subdata_d["noind"].sort_values([0,1]).to_numpy()
            del subdata_d[industry]
            
            # daily_local = daily_MCAP.to_numpy()
        distinct_int_dates = np.unique(quarterly_local[:,columns_local_q["int_date"]])
        if industry != "noind":
            distinct_int_dates = distinct_int_dates[distinct_int_dates>int(industry.split("_")[1])]
        
        progress_bar.change_bar(positions = [1],n=[max(1,len(distinct_int_dates))],name = ["Ind "+industry])
        
        for distinct_date in distinct_int_dates:
            # b_time.append(time.time())
            # distinct_date = distinct_int_dates[2500]
            
            ## cut out relevant data
            quarterly_local = quarterly_local[quarterly_local[:,columns_local_q["int_date"]]>distinct_date-100,:]
            local_data = quarterly_local[quarterly_local[:,columns_local_q["int_date"]]<=distinct_date,:]
            # local_data = local_data[local_data[:,columns_local_q["int_date"]]>distinct_date-100,:]
            n_reports = (local_data[:,columns_local_q["int_date"]]==distinct_date).sum()
            # sum_n_reports += n_reports
            
            ## reverse sort by date
            local_data = local_data[local_data[:, columns_local_q["date"]].argsort()[::-1]]
            
            ## keep unique index
            gvkeys_local, index = np.unique(local_data[:, columns_local_q["gvkey"]], return_index=True)
            local_data =  local_data[index,:]
            
            ## gain values for market cap
            # a_time = time.time()
            daily_local = daily_local[daily_local[:,columns_local_d["int_date"]]>distinct_date-100,:]
            local_daily = daily_local[daily_local[:,columns_local_d["int_date"]]<=distinct_date,:]
            # local_daily = local_daily[local_daily[:,columns_local_d["int_date"]]>distinct_date-100,:]
            local_daily = local_daily[is_in_set_pnb(
                np.array(local_daily[:, columns_local_d["gvkey"]],dtype=np.int64), 
                np.array(gvkeys_local,dtype=np.int64))]
            local_daily = local_daily[np.unique(local_daily[:, columns_local_d["gvkey"]], return_index=True)[1],:]
            # print("{:9.7f}".format(time.time()-a_time))
            
            ## merging the two:
            gvkeys_join = np.intersect1d(local_data[:, columns_local_q["gvkey"]],
                                         local_daily[:, columns_local_d["gvkey"]])
            
            ## left join on gvkey
            intersection_bools = np.isin(local_data[:, columns_local_q["gvkey"]], gvkeys_join)
            local_data_intersection_index = np.array([
                intersection_bools,
                np.arange(len(intersection_bools)),
                local_data[:,columns_local_d["gvkey"]]]).T
            local_data_intersection_index = local_data_intersection_index[
                local_data_intersection_index[:,0]==True]
            local_data_intersection_index = local_data_intersection_index[
                local_data_intersection_index[:, 2].argsort()]
            
            # local_data = local_data[np.isin(local_data[:, columns_local_q["gvkey"]], gvkeys_join)]
            # local_data = local_data[local_data[:, columns_local_q["gvkey"]].argsort()]
            
            ## cut out joined rows from daily_data and sort
            local_daily = local_daily[np.isin(local_daily[:, columns_local_d["gvkey"]], gvkeys_join)]
            local_daily = local_daily[local_daily[:, columns_local_d["gvkey"]].argsort()]
            
            
            
            ## check if at least five companies are available, otherwise shove them into the no_ind category
            ## if the category is noind, skip this step
            if local_data.shape[0]< min_companies_per_industry_month:
                if industry != "noind":
                    subdata_q["noind"].append(local_data)
                    subdata_d["noind"].append(local_daily)
                continue
                    
            ## stack numpy arrays
            merged_array = np.hstack((local_data, np.zeros((local_data.shape[0],1))))
            merged_array[local_data_intersection_index[:,1].tolist(),[columns_local_q["MCAP"]]] = \
                local_daily[:, 2]
            cutout_rows = merged_array[:,columns_local_q["int_date"]]==distinct_date
            if sum(cutout_rows) == 0:
                print("WARNING: THIS SHOULD NOT HAPPEN.")
                continue
            else:
                skipcounter+=(n_reports-sum(cutout_rows))
            
            appendix = np.zeros(shape = (sum(cutout_rows),len(columns_ind_q)))
            
            ## IDCE	Industry adjusted % change in capital expenditures 
            mcapsum = np.nansum(merged_array[:,columns_local_q["MCAP"]])
            mcapsum = max(1,mcapsum)
            appendix[:,0] = \
                np.nansum(merged_array[:,columns_local_q["DCE"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum
            
            ## IDE	Industry-adjusted change in employees 
            appendix[:,1] = \
                np.nansum(merged_array[:,columns_local_q["DE"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum


            ## IDPM	Industry-adjusted change in profit margin 
            appendix[:,2] = \
                np.nansum(merged_array[:,columns_local_q["IDPM"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## ISS	Industry sales share
            appendix[:,4] = \
                np.nansum(merged_array[:,columns_local_q["SP"]])
            if appendix[:,4].sum() == 0:
                appendix[:,4] = 1
            
            ## ISCH	Industry sales concentration 
            appendix[:,3] = np.nansum((merged_array[:,columns_local_q["SP"]]/appendix[0,4])**2)
            
            ## IACA	Industry adjusted absolute accruals
            appendix[:,5] = \
                np.nansum(merged_array[:,columns_local_q["ACA"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## IROI	industry adjusted return on investment
            appendix[:,6] = \
                np.nansum(merged_array[:,columns_local_q["ROI"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## IROE	industry adjusted return on equity
            appendix[:,7] = \
                np.nansum(merged_array[:,columns_local_q["ROE"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## IOP industry adjusted operating profitability
            appendix[:,8] = \
                np.nansum(merged_array[:,columns_local_q["OP"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## IRDS	industry adjusted research and development to sales
            appendix[:,9] = \
                np.nansum(merged_array[:,columns_local_q["RDS"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## IDS industry adjusted change in sales
            appendix[:,10] = \
                np.nansum(merged_array[:,columns_local_q["DS"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## IFSS	industry adjusted financial statement score
            appendix[:,11] = \
                np.nansum(merged_array[:,columns_local_q["FSS"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            
            ## IDEAR industry adusted change in earnings
            appendix[:,12] = \
                np.nansum(merged_array[:,columns_local_q["DEAR"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum

            appendix[:,13] = \
                np.nansum(merged_array[:,columns_local_q["IDAT"]]*merged_array[:,columns_local_q["MCAP"]])/\
                    mcapsum


            ## cut out data from distinct date only:
            merged_array = merged_array[:,cutout_columns]
            merged_array = np.hstack((merged_array[cutout_rows,:],appendix))
            ind_data.append(merged_array)
            progress_bar([1])
            # print("b_time",time.time()-b_time[-1])
        progress_bar([0])
    ind_data_con = np.concatenate(ind_data)
    ind_data_con = pd.DataFrame(ind_data_con,columns = ["gvkey","date",*columns_ind_q])
    ind_data_con.set_index(["gvkey","date"],inplace=True)
    for column in ind_data_con.columns:
        ind_data_con[column] = ind_data_con[column].astype(np.float64)   
    
    ## IDPM needs to be renamed
    # overlap_columns = set(ind_data_con.columns.tolist()).difference(set(quarterly.columns.tolist()))
    ind_data_con.rename(columns = {"IDPM":"ind_IDPM","IDAT":"ind_IDAT"},inplace=True)
    
    ## save backup of industrial demeaning
    ind_data_con.to_hdf(PATH_DATA+"ind_q_raw_pre_noind.h5", key="data")    
    
    
    
    ind_data_con = pd.read_hdf(PATH_DATA+"ind_q_raw_pre_noind.h5", key = "data")    
    quarterly = pd.read_hdf(PATH_DATA+"quarterly_P3_s2.h5",key = "data")
    quarterly = pd.merge(
        pd.read_hdf(PATH_DATA+"quarterly_to_monthly.h5","data")["SP"], quarterly,
        left_index=True,right_on = ["gvkey","date"],how="inner")
    quarterly.reset_index(inplace=True)
    quarterly["date"] = quarterly["date"].astype(int).apply(lambda x: dt.datetime.strptime(str(x),"%Y%m%d")).dt.date
    quarterly.set_index(["gvkey","date"],inplace=True)
    
    quarterly = pd.merge(quarterly, ind_data_con,left_index = True, right_index = True,
                         how = "left")
    
    ## IDAT	    Industry-adjusted change in asset turnover 
    quarterly["IDAT"] = quarterly["IDAT"]-quarterly["ind_IDAT"]
    del quarterly["ind_IDAT"]
    '''
    #########################
    ########## IDAT #########
    #########################

              NA    ZERO    mean     std    skew     kurt
    IDAT  0.3212  0.0001  0.0173  0.2347  4.1067  42.9083
                    JB   99.5%    0.5%    min    max  median
    IDAT  6.539093e+06  1.4166 -0.7284 -1.958  6.359  0.0038
    '''
    
    ## IDCE	    Industry adjusted % change in capital expenditures 
    quarterly["IDCE"] = quarterly["DCE"]-quarterly["IDCE"] 
    '''
    #########################
    ########## IDCE #########
    #########################

              NA    ZERO    mean     std   skew     kurt
    IDCE  0.4035  0.0001 -0.0122  1.6681  0.511  21.7695
                   JB  99.5%    0.5%      min      max  median
    IDCE  1206740.694  9.919 -9.8647 -17.5183  17.8833 -0.0355
    '''    
    
    ## IDE	    Industry-adjusted change in employees 
    quarterly["IDE"] = quarterly["DE"]-quarterly["IDE"]
    '''
    #########################
    ########## IDE ##########
    #########################

             NA    ZERO   mean     std    skew    kurt
    IDE  0.0753  0.0112 -0.714  2.4166 -0.2928  5.0598
                  JB   99.5%    0.5%      min      max  median
    IDE  146343.7245  8.0038 -9.2659 -19.8302  19.5138 -0.2206
    '''
    
    ## IDPM	    Industry-adjusted change in profit margin 
    quarterly["IDPM"] = quarterly["IDPM"]-quarterly["ind_IDPM"]
    del quarterly["ind_IDPM"]
    '''
    #########################
    ########## IDPM #########
    #########################

           NA    ZERO   mean     std    skew      kurt
    IDPM  0.0  0.0243  0.002  0.4043  2.0612  538.4684
                    JB   99.5%    0.5%      min      max  median
    IDPM  3.364717e+07  1.2292 -1.0663 -20.0212  20.0328 -0.0006
    '''
    
    ## ISCH	    Industry sales concentration 
    if False:
        quarterly["ISCH"]
    '''
    #########################
    ########## ISCH #########
    #########################

           NA    ZERO   mean    std    skew     kurt
    ISCH  0.0  0.0002  0.084  0.099  3.8075  20.0445
                    JB   99.5%    0.5%  min  max  median
    ISCH  4.569893e+06  0.7286  0.0098  0.0  1.0  0.0529
    '''
    
    ## ISS	    Industry sales share
    quarterly["ISS"] = quarterly["SP"] / quarterly["ISS"] 
    '''
    #########################
    ########## ISS ##########
    #########################

             NA    ZERO    mean     std    skew     kurt
    ISS  0.1583  0.0233  0.0127  0.0454  8.2718  93.0802
                   JB   99.5%  0.5%     min  max  median
    ISS  2.215571e+07  0.3178   0.0 -0.2899  1.0  0.0008
    '''
    
    ## IACA	    Industry adjusted absolute accruals
    quarterly["IACA"] = quarterly["ACA"]-quarterly["IACA"]
    '''
    #########################
    ########## IACA #########
    #########################

              NA    ZERO   mean     std    skew     kurt
    IACA  0.2634  0.0002  0.016  0.0762  6.3139  77.4612
                    JB   99.5%    0.5%     min     max  median
    IACA  1.424737e+07  0.4975 -0.0804 -0.5597  2.9008 -0.0011
    '''
    
    ## IROI	    industry adjusted return on investment
    quarterly["IROI"] = quarterly["ROI"]-quarterly["IROI"]
    '''
    #########################
    ########## IROI #########
    #########################

              NA    ZERO    mean     std     skew      kurt
    IROI  0.2769  0.0009 -0.3207  1.8127 -10.4333  144.3443
                    JB   99.5%     0.5%      min     max  median
    IROI  3.512822e+07  1.4792 -12.4275 -51.4319  7.6642  -0.037
    '''
    
    ## IROE	    industry adjusted return on equity
    quarterly["IROE"] = quarterly["ROE"]-quarterly["IROE"]
    '''
    #########################
    ########## IROE #########
    #########################

             NA    ZERO    mean     std    skew     kurt
    IROE  0.254  0.0001 -0.1472  1.6392 -0.5096  55.4232
                    JB   99.5%    0.5%      min      max  median
    IROE  3.256160e+06  8.3384 -9.2183 -29.6391  27.9998 -0.0466
    '''
    
    ## IOP	    industry adjusted operating profitability
    quarterly["IOP"] = quarterly["OP"]-quarterly["IOP"]
    '''
    #########################
    ########## IOP ##########
    #########################

             NA   ZERO   mean     std     skew      kurt
    IOP  0.1816  0.004 -0.027  0.2345  15.0181  622.8563
                   JB   99.5%    0.5%     min   max  median
    IOP  9.270049e+07  0.8121 -0.4673 -0.9893  11.0 -0.0549
    '''
    
    ## IRDS	    industry adjusted research and development to sales
    quarterly["IRDS"] = quarterly["RDS"]-quarterly["IRDS"]
    '''
    #########################
    ########## IRDS #########
    #########################

              NA    ZERO    mean     std    skew     kurt
    IRDS  0.1816  0.0148  0.2799  1.3626  5.4891  32.6629
                    JB   99.5%    0.5%      min   max  median
    IRDS  9.147138e+06  9.5117 -0.7373 -11.1015  10.0 -0.0027
    '''
    
    ## IDS	    industry adjusted change in sales
    quarterly["IDS"] = quarterly["DS"]-quarterly["IDS"]
    '''
    #########################
    ########## IDS ##########
    #########################

             NA    ZERO    mean     std     skew      kurt
    IDS  0.1816  0.0005  0.0102  0.5546  12.6941  219.2488
                   JB   99.5%    0.5%      min      max  median
    IDS  5.242841e+07  2.2159 -1.0435 -10.7044  10.9964 -0.0101
    '''
    
    
    ## IFSS	    industry adjusted financial statement score
    quarterly["IFSS"] = quarterly["FSS"]-quarterly["IFSS"]
    '''
    #########################
    ########## IFSS #########
    #########################

           NA    ZERO    mean     std   skew    kurt
    IFSS  0.0  0.0038 -0.7156  1.9605  0.141  0.2474
                   JB  99.5%    0.5%     min  max  median
    IFSS -162801.9579    5.0 -5.3508 -8.2627  9.0 -0.8096
    '''
    
    ## IDEAR	industry adusted change in earnings
    quarterly["IDEAR"] = quarterly["DEAR"]-quarterly["IDEAR"]
    """
    #########################
    ######### IDEAR #########
    #########################

              NA    ZERO   mean     std    skew     kurt
    IDEAR  0.186  0.0004  0.007  1.5137  0.1643  25.8419
                     JB   99.5%    0.5%      min      max  median
    IDEAR  1.397784e+06  9.7726 -9.6504 -15.5685  15.5781  0.0018
    """

    # columns_ind_q = ["IDCE", "IDE", "IDPM", "ISCH", "ISS", "IACA", "IROI", "IROE", "IOP", "IRDS", 
    #            "IDS", "IFSS", "IDEAR", "IDAT"]
    
    
    columns_q_total = [
        "ACA", "ACV", "ACW", "AG1Y", "C", "CEI", "CFD", "CFV", "CIN", "CP", "CR", "DCE", "DCR", "DCSE", 
        "DD", "DE", "DGMDS", "DI", "DLTD", "DNOA", "DPPE", "DQR", "DRD", "DS", "DSDAR", "DSDI", "DSDSGA", 
        "DSI", "DTAX", "EIS", "EV", "FSS", "GP", "IDAT", "IDCE", "IDE", "IDPM", "ISCH", "L", 
        "OC", "OP", "QR", "RDS", "RE", "ROA", "ROE", "ROI", "RS", "SC", "SD", "SI", "SR", "TANG", "TIBI", 
        "DEAR", "RDE", "RDC", "SGAE", "ARI", "DSGAE", "CFRD", "DACA", "DROI", "DROE", "DRDS", "DFSS", 
        "IACA", "IROI", "IROE", "IOP", "IRDS", "IDS", "IFSS", "IDEAR", "ISS"]
    
    ########################################
    ################# FSS2 #################
    ########################################
    
    quarterly["FSS2"] = quarterly[columns_q_total].isna().sum(axis=1)
    quarterly["FSS2"] += (quarterly[columns_q_total]==0).sum(axis=1)
    quarterly["FSS2"] /= quarterly.shape[1]
    columns_q_total.append("FSS2")
    
    '''
    #########################
    ########## FSS2 #########
    #########################

           NA    ZERO    mean     std    skew    kurt
    FSS2  0.0  0.0239  0.3065  0.3253  0.9274 -0.7225
                  JB   99.5%  0.5%  min     max  median
    FSS2 -17186.8395  0.9351   0.0  0.0  0.9481  0.1429
    
    ratio_60 = (quarterly["FSS2"]>.6).sum()/quarterly.shape[0]
    ratio_40 = (quarterly["FSS2"]>.4).sum()/quarterly.shape[0]
    
    Rows above 60% na for the columns:
        22.88%
    
    Rows above 40% na:
        29,99%
    '''    
    
    quarterly = quarterly[~quarterly.index.duplicated(keep="first")]
    
    ## prepare date column.
    # quarterly.reset_index(inplace=True)
    # # quarterly["date"] = quarterly["date"].astype('<M8[ns]')
    # quarterly["date"] = quarterly["date"].astype('datetime64[ns]')
    # quarterly.set_index(["gvkey","date"],inplace=True)
    
    quarterly.to_hdf(PATH_DATA+"quarterly_P3.h5",key = "data")
    
        
####################################################################################################
####################################################################################################
####################################  Linkage and partitioning  ####################################
####################################################################################################
####################################################################################################  
        
if ("FULL" in mode or "linkeage" in mode):
    
    #### load index data only
    quarterly = pd.read_hdf(PATH_DATA+"quarterly_P3.h5",key = "data")[["FSS2"]]
    monthly = pd.read_hdf(PATH_DATA+"monthly_P3.h5",key = "data")[[]]
    weekly = pd.read_hdf(PATH_DATA+"weekly_P3.h5",key = "data")[["mcap"]]
    
    #### expand metadata
    quarterly["q_date"] = quarterly.index.get_level_values("date")
    quarterly.reset_index(inplace=True)
    quarterly["date"] = quarterly["q_date"]
    quarterly.set_index(["gvkey","date"],inplace=True)
    # quarterly["q"] = True
    quarterly["q_count"] = 1
    quarterly["q_count"] = quarterly.groupby("gvkey")["q_count"].cumsum()
    
    ## extract and prepare metadata for monthly and weekly
    monthly["m_date"] = monthly.index.get_level_values("date")
    monthly["m"] = True
    weekly["w_date"] = weekly.index.get_level_values("date")
    weekly["w"] = True
    
    #### merge and ffill
    ## quarterly
    weekly = pd.merge(weekly,quarterly,left_on = ["gvkey","date"], right_on=["gvkey","date"],
                      how="outer")
    del quarterly
    weekly.sort_index(inplace=True)
    weekly["q_date"] = weekly["q_date"].groupby("gvkey").ffill()
    weekly["q_count"] = weekly["q_count"].groupby("gvkey").ffill()
    weekly["FSS2"] = weekly["FSS2"].groupby("gvkey").ffill()
    
    weekly = weekly[weekly["w"] == True]
    ## cut out weekly observations without quarterly statement:
    weekly = weekly[~weekly["q_date"].isna()]
    ## calculate distance of weekly dates to quarterly dates
    weekly["q_dist"] = (weekly["w_date"]-weekly["q_date"]).apply(lambda x: x.days)#.dt.days
    
    # ## cut out weekly observations without quarterly statement:
    # weekly = weekly[~weekly["q_dist"].isna()]
    '''
    Cutting out non-quarterly matched observations:
        Reduction from 12,469,242 to 11,695,528 observations
    '''
    weekly = weekly[weekly["q_dist"]<200]
    '''
    Cutting out observations where the last quarterly report is at least 200 days ago
        Reduction from 11,695,528 to 11,678,362
    
    Count of observations with at least 
        4 quarters:
            11,347,036
        6 quarters:
            11,076,820
        8 quarters:
            10,718,862
        12 quarters:
            9,862,146
    Good!
    '''
    weekly["q_dist"].hist(bins=20,rwidth=0.7)
    plt.ylabel("N obs") 
    plt.xlabel("Distance to last quarterly report")
    plt.title("Date distance of weekly observations to last quarter")
    plt.show()
    plt.savefig(PATH_DATA+"q_dist.png")
    
    ## monthly
    weekly = pd.merge(weekly,monthly,left_index=True, right_index=True,
                      how="outer")
    del monthly
    weekly.sort_index(inplace=True)
    
    weekly["m_date"] = weekly["m_date"].groupby("gvkey").ffill()
    weekly["m_count"] = 0
    weekly.loc[weekly["m"] ==True,"m_count"] = 1
    weekly["m_count"] = weekly.groupby("gvkey")["m_count"].cumsum()
    
    weekly = weekly[weekly["w"] == True]
    
    ## cut out weekly observations without monthly observation:
    weekly = weekly[~weekly["m_date"].isna()]
    
    ## calculate distance of weekly dates to monthly dates
    weekly["m_dist"] = (weekly["w_date"]-weekly["m_date"]).apply(lambda x: x.days)

    # weekly = weekly[~weekly["m_dist"].isna()]
    '''
    Cutting out non-quarterly matched observations:
        Reduction from 11,678,362 to 11,640,799 observations
    '''
    weekly = weekly[weekly["m_dist"]<32]
    '''
    Cutting out observations where the last monthly observation is at least 32 days ago
        Reduction from 11,646,468 to 11,642,127
    
    Count of observations with at least 
        6 months:
            11,259,646
        12 months:
            10,720,847
        18 months:
            10,193,389
        24 months:
            9,695,987
    '''
    weekly["m_dist"].hist(bins=10,rwidth=0.7)
    plt.ylabel("N obs") 
    plt.xlabel("Distance to last monthly obs")
    plt.title("Date distance of weekly observations to last month")
    plt.show()
    plt.savefig(PATH_DATA+"m_dist.png")
    
    weekly["w_count"] = 1
    weekly["w_count"] = weekly.groupby("gvkey")["w_count"].cumsum()
    '''
    Count of observations with at least 
        6 weeks:
            11,516,908
        13 weeks:
            11,349,943
        26 months:
            11,042,480
    '''
    del weekly["m"] ,weekly["w"]
    weekly["q_dist"]/=91 ## base the q_dist on a fourth of the year as expected average distance between quarterly reports
    weekly["m_dist"]/=31 ## base the m_dist on maximum length of a month.
    weekly.to_hdf(PATH_DATA+"linkeage.h5",key="data")
    del weekly
    
if ("FULL" in mode or "partitioning" in mode):
    label_file = "labels{}.csv"
    map_file = "maps{}.csv"
    data_identification_file = "ident{}.csv"
    x_name = "data_b{batch:d}x{freq}.h5"
    y_name = "data_b{batch:d}y1.h5"
    
    '''
    Now here the dataset is really large, we have over 11 million observations with
    140 predictors each and 6 dependent variables. For testing and reduced ressource setups, 
    it is valuable having a smaller dataset only keeping the most relevant datapoints.
    
    The following block creates a modified linkeage table that only takes the largest 500
    companies each week, representing the S&P500 (kind off). We do this because unfortunately
    we do not have access to historical S&P500 components prior to 1993, which we would need 
    for training purposes.
    '''
    
    #### splitting up data and creating linkeage 
    linkeage_table = pd.read_hdf(PATH_DATA+"linkeage.h5",key="data")
    if not os.path.exists(PATH_DATA+"linkeage_f500.h5"):
        linkeage_table_f500 = linkeage_table.groupby("date")["mcap"].nlargest(500).droplevel(0)
        linkeage_table_f500 = pd.merge(
            linkeage_table_f500.to_frame("mcap")[[]],linkeage_table, 
            left_index=True,right_index=True,how="inner")
        linkeage_table_f500 = linkeage_table_f500[~linkeage_table_f500.index.duplicated(keep='first')]
        linkeage_table_f500.to_hdf(PATH_DATA+"linkeage_f500.h5",key="data")
    else:
        linkeage_table_f500= pd.read_hdf(PATH_DATA+"linkeage_f500.h5",key="data")
    
    '''
    We create 2 seperate datasets:
        A full sample dataset including all over 12 million weekly observations
        An S&P500 dataset with just 1.500.000 observations
    
    '''
    for linkeage_, dataset_name_ in [[linkeage_table,"complete"],[linkeage_table_f500,"f500"]]:
        # linkeage_, dataset_name_ = linkeage_table_f500.copy(),"f500"
        
        sub_path = PATH_DATA+dataset_name_+"/"
        
        ## check if directory exists
        if not os.path.exists(PATH_DATA+dataset_name_):
            os.mkdir(PATH_DATA+dataset_name_) 
            
        ##################################################
        ############ Weekly data and linkeage ############
        ##################################################
        '''
        Saving by month.
        '''
        weekly_data = pd.read_hdf(PATH_DATA+"weekly_P3.h5")
        target_columns = ["ret1w","ret4w","ret13w","rvarf1w","rvarf4w","rvarf13w"]
        weekly_columns = weekly_data.columns.tolist()
        weekly_columns = [column for column in weekly_columns if column not in target_columns]
        
        linkeage_columns = [column for column in linkeage_.columns \
                            if column not in weekly_data.columns]
        weekly_data = pd.merge(weekly_data,linkeage_[linkeage_columns],
                               left_index=True, right_index=True,how="inner")        
        weekly_data = weekly_data[~weekly_data.index.duplicated(keep='first')]
        weekly_data["year"] = weekly_data["w_date"].apply(lambda x: x.year)
        weekly_data["month"] = weekly_data["w_date"].apply(lambda x: x.month)
        
        ## linkeage also now contains the target
        linkeage_columns = [*linkeage_.columns,*target_columns]
        linkeage_copy = weekly_data[[*linkeage_columns,"year","month",]]
        
        weekly_data = weekly_data[[*weekly_columns,"year","month"]]
        weekly_data = {x:y for x,y in weekly_data.groupby(["year","month"], as_index=False)}
        linkeage_copy = {x:y for x,y in linkeage_copy.groupby(["year","month"], as_index=False)}
        
        
        pd.DataFrame(columns = ["name"]).to_csv(sub_path+map_file.format("w"),sep=";")
        pd.DataFrame(weekly_columns,columns = ["name"]).\
            to_csv(sub_path+label_file.format("w"),sep=";")
        
        ident_file = []
        file_index = 0
        
        #### save linkeage- and meta-data by month
        month_indices = list(weekly_data.keys())
        month_indices.sort()
        
        ## ignore warning regarding performance of hdf files.
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        for month_index in month_indices:
            # month_index = (2000,11)
            print(month_index,end=" ,  ")
            ## file location tracker: ident_file:
            input_target_date = dt.datetime(*month_index,28)+dt.timedelta(days=4)
            input_target_date = input_target_date-dt.timedelta(days=input_target_date.day)
            n_obs = weekly_data[month_index].shape[0]
            ident_file.append([input_target_date,input_target_date,file_index,n_obs])
            
            weekly_data[month_index].sort_index(inplace=True)
            linkeage_copy[month_index].sort_index(inplace=True)
            
            weekly_data[month_index][weekly_columns].to_hdf(sub_path+x_name.format(
                batch = file_index, freq="w"),key="data")
            
            linkeage_copy[month_index][linkeage_columns].to_hdf(sub_path+y_name.format(
                batch = file_index),key="data")
            
            del weekly_data[month_index], linkeage_copy[month_index]
            file_index+=1
        
        ident_file = pd.DataFrame(ident_file,columns = ["input_date","target_date","batch","n_obs"])
        ident_file.to_csv(sub_path+data_identification_file.format("w"),sep=";")
        
        del weekly_data, linkeage_copy, ident_file
        
        ##################################################
        ################## Monthly data ##################
        ##################################################
        '''
        monthly data is being saved by the year. Upon import of data, the lookback windows are created and
        the observations are linked. Because of this, there is no value in precutting the data and saving 
        a ton of copies.
        
        Solution:
            we save months AND quarters by the year and do the puzzling acrobatics upon loading the data for 
            the models.
        '''
        ## import monthly file
        monthly_data = pd.read_hdf(PATH_DATA+"monthly_P3.h5")
        monthly_data = monthly_data[~monthly_data.index.duplicated(keep="first")]
        columns_monthly = ["BM", "CFP", "DIVP", "DM6M", "EP", "IBM", "ICFP", "IM6M", 
                           "IM12M", "IMCAP", "M1M", "M6M", "M12M", "M36M", "MCAP", "SP", 
                           "RVAR1M", "RVAR6M", "DEP1M", "DEP12M", "IEP", "DCFP1M", "DCFP12M", 
                           "DSP1M", "DSP12M", "ISP"]

        linkeage_copy_m = linkeage_.copy()
        linkeage_copy_m = linkeage_copy_m[["m_date"]]
        linkeage_copy_m.reset_index(inplace=True)
        del linkeage_copy_m["date"]
        linkeage_copy_m = linkeage_copy_m.rename(columns = {"m_date":"date"})
        # linkeage_copy_m.drop_duplicates(inplace=True)
        # linkeage_copy_m["year"] = linkeage_copy_m["date"].dt.year
        linkeage_copy_m.set_index(["gvkey","date"],inplace=True)
        linkeage_copy_m = linkeage_copy_m[~linkeage_copy_m.index.get_level_values("gvkey").duplicated()]
        
        monthly_columns = monthly_data.columns
        index_cols = monthly_data.index.names
        monthly_data = pd.merge(
            monthly_data.reset_index(),linkeage_copy_m,
            left_on=["gvkey"], right_on=["gvkey"], how= "inner").set_index(index_cols)
        monthly_data["year"] = monthly_data.index.get_level_values("date")
        monthly_data["year"] = monthly_data["year"].apply(lambda x: x.year)
        ## problem: this should include the full history of monthly data
        monthly_data = monthly_data[~monthly_data.index.duplicated(keep='last')]
        monthly_data = {x[0]:y for x,y in monthly_data.groupby(["year"], as_index=False)}
        
        del linkeage_copy_m
        
        pd.DataFrame(columns = ["name"]).to_csv(sub_path+map_file.format("m"),sep=";")
        pd.DataFrame(monthly_columns,columns = ["name"]).\
            to_csv(sub_path+label_file.format("m"),sep=";")
        
        #### save monthly data by year
        ident_file = []
        file_index = 0
        
        year_indices = list(monthly_data.keys())
        year_indices.sort()
        
        for year_index in year_indices:
            # file_name = 'data_b87y1.h5'
            
            print(year_index,end=" ,  ")
            
            input_target_date = dt.datetime(year_index,12,31)
            n_obs = monthly_data[year_index].shape[0]
            ident_file.append([input_target_date,input_target_date,file_index,n_obs])
            
            ## save monthly data
            monthly_data[year_index][columns_monthly].to_hdf(sub_path+x_name.format(
                batch = file_index, freq="m"),key="data")
            
            file_index +=1
            del monthly_data[year_index]
        
        ident_file = pd.DataFrame(ident_file,columns = ["input_date","target_date","batch","n_obs"])
        ident_file.to_csv(sub_path+data_identification_file.format("m"),sep=";")
        
        del monthly_data, ident_file
        
        ##################################################
        ################# Quarterly data #################
        ##################################################
        
        ## import quarterly file
        quarterly_data = pd.read_hdf(PATH_DATA+"quarterly_P3.h5")
        quarterly_data = quarterly_data[~quarterly_data.index.duplicated(keep="first")]
        columns_quarterly = [
            "ACA", "ACV", "ACW", "AG1Y", "C", "CEI", "CFD", "CFV", "CIN", "CP", "CR", "DCE", 
            "DCR", "DCSE", "DD", "DE", "DGMDS", "DI", "DLTD", "DNOA", "DPPE", "DQR", "DRD", 
            "DS", "DSDAR", "DSDI", "DSDSGA", "DSI", "DTAX", "EIS", "EV", "FSS", "FSS2", "GP", 
            "IDAT", "IDCE", "IDE", "IDPM", "ISCH", "L", "OC", "OP", "QR", "RDS", "RE", "ROA", 
            "ROE", "ROI", "RS", "SC", "SD", "SI", "SR", "TANG", "TIBI", "DEAR", "RDE", "RDC", 
            "SGAE", "ARI", "DSGAE", "CFRD", "DACA", "DROI", "DROE", "DRDS", "DFSS", "IACA", 
            "IROI", "IROE", "IOP", "IRDS", "IDS", "IFSS", "IDEAR", "ISS"]
        
        linkeage_copy_q = linkeage_.copy()
        linkeage_copy_q = linkeage_copy_q[["q_date"]]
        linkeage_copy_q.reset_index(inplace=True)
        del linkeage_copy_q["date"]
        linkeage_copy_q = linkeage_copy_q.rename(columns = {"q_date":"date"})
        # linkeage_copy_q.drop_duplicates(inplace=True)
        # linkeage_copy_q["year"] = linkeage_copy_q["date"].dt.year
        linkeage_copy_q.set_index(["gvkey","date"],inplace=True)
        linkeage_copy_q = linkeage_copy_q[~linkeage_copy_q.index.get_level_values("gvkey").duplicated()]
        
        quarterly_columns = quarterly_data.columns
        index_cols = quarterly_data.index.names
        quarterly_data = pd.merge(quarterly_data.reset_index(),linkeage_copy_q,
                                left_on=["gvkey"], right_on=["gvkey"], how= "inner").set_index(index_cols)
        quarterly_data["year"] = quarterly_data.index.get_level_values("date")
        quarterly_data["year"] = quarterly_data["year"].apply(lambda x: x.year)
        quarterly_data = quarterly_data[~quarterly_data.index.duplicated(keep='last')]
        quarterly_data = {x[0]:y for x,y in quarterly_data.groupby(["year"], as_index=False)}
        
        del linkeage_copy_q
        
        pd.DataFrame(columns = ["name"]).to_csv(sub_path+map_file.format("q"),sep=";")
        pd.DataFrame(quarterly_columns,columns = ["name"]).\
            to_csv(sub_path+label_file.format("q"),sep=";")
        
        #### save quarterly data by year
        ident_file = []
        file_index = 0
        
        year_indices = list(quarterly_data.keys())
        year_indices.sort()
        
        for year_index in year_indices:
            
            print(year_index,end=" ,  ")
            
            input_target_date = dt.datetime(year_index,12,31)
            n_obs = quarterly_data[year_index].shape[0]
            ident_file.append([input_target_date,input_target_date,file_index,n_obs])
            
            ## save quarterly data
            quarterly_data[year_index][columns_quarterly].to_hdf(sub_path+x_name.format(
                batch = file_index, freq="q"),key="data")
            
            file_index +=1
            del quarterly_data[year_index]
        
        ident_file = pd.DataFrame(ident_file,columns = ["input_date","target_date","batch","n_obs"])
        ident_file.to_csv(sub_path+data_identification_file.format("q"),sep=";")
        
        del quarterly_data, ident_file
       
####################################################################################################
####################################################################################################
####################################################################################################
#########################   Data for Common Factor Models and Benchmarks   #########################
####################################################################################################
####################################################################################################
####################################################################################################
        
### Step 1: dataset extraction ###
if False:
    columns_input = ["rvar1w","rvar4w","rvar13w","rq1w","rq4w","rq13w"]
    columns_output = ["rvarf1w","rvarf4w","rvarf13w"]
    phd3_dataset_variants = ["complete","f500"]
    phd3_PATH = PATH_DATA+"{variant:s}/"
    
    data_identification_file = "ident{}.csv"
    x_name = "data_b{batch:d}x{freq}.h5"
    y_name = "data_b{batch:d}y1.h5"
    
    ### !!! semivariance?
    
    for variant in phd3_dataset_variants:
        # variant = "f500"
        ## empty path 
        base_path = phd3_PATH.format(variant = variant)
        input_data = []
        output_data = []
        
        ## load ident file
        batches = pd.read_csv(
            base_path + data_identification_file.format("w"),
            sep = ";")["batch"].tolist()
        
        ## load batches
        for batch in batches:
            input_data.append(
                pd.read_hdf(base_path + x_name.format(
                    batch = batch, freq = "w"), key="data"))
            output_data.append(
                pd.read_hdf(base_path + y_name.format(
                    batch = batch), key="data")[
                        columns_output])
        input_data = pd.concat(input_data)
        for frequency in [1,4,13]:
            input_data[f"rq{frequency}w"] = input_data[f"rq{frequency}w"]**.5*\
                input_data[f"rvar{frequency}w"]
        
        
        
        input_data[columns_input].to_hdf(phd3_PATH.format(variant=f"{variant}_benchmark")+ \
                                         f"HARx_{variant:s}.h5",key="data")
        input_data.to_hdf(phd3_PATH.format(variant=f"{variant}_benchmark")+ \
                          f"ALLWx_{variant:s}.h5",key="data")
        
        pd.concat(output_data).to_hdf(phd3_PATH.format(variant=f"{variant}_benchmark")+ \
                                      f"HARy_{variant:s}.h5",key="data")
        del input_data, output_data
        
        
        
################################################################################
################################################################################
#############  Saving isolated mcap series, ff3, and ff5+m series.  ############
################################################################################
################################################################################
if False:
    columns_input = {
        "w":["b","mcap"],
        "m":["BM","M12M"],
        "q":["CIN","OP"]}
    columns_ff3 = ["b","mcap","BM"]
    columns_ff5M = ["b","mcap","M12M","BM","CIN","OP"]
    
    columns_output_import = ["ret1w","ret4w","ret13w","m_date","q_date"]
    columns_output = ["ret1w","ret4w","ret13w"]
    phd3_dataset_variants = ["complete","f500"]
    phd3_PATH = PATH_DATA+"{variant:s}/"
    
    data_identification_file = "ident{}.csv"
    x_name = "data_b{batch:d}x{freq}.h5"
    y_name = "data_b{batch:d}y1.h5"
    
    for variant in phd3_dataset_variants:
        # variant = "f500"
        
        #### load files for weekly monthly and quarterly data
        
        ## empty path 
        base_path = phd3_PATH.format(variant = variant)
        
        ## empty datasets
        input_data = {"w":[],"m":[],"q":[]}
        output_data = []
        
        ## load ident file
        for frequency in ["w","m","q"]:
            # frequency = "q"
            batches = pd.read_csv(
                base_path + data_identification_file.format(frequency),
                sep = ";")["batch"].tolist()
        
            for batch in batches:
                input_data[frequency].append(
                    pd.read_hdf(base_path + x_name.format(
                        batch = batch, freq = frequency), key="data")[columns_input[frequency]])
                if frequency == "w":
                    output_data.append(
                        pd.read_hdf(base_path + y_name.format(
                            batch = batch), key="data"))
            if frequency == "w":
                output_data = pd.concat(output_data)
            input_data[frequency] = pd.concat(input_data[frequency])
            
        #### merge the frequencies
        
        input_data["w"][["m_date","q_date"]] = output_data[["m_date","q_date"]]
        del output_data["m_date"], output_data["q_date"]
        
        input_data["w"] = pd.merge(input_data["w"].reset_index(),input_data["m"], left_on = ["gvkey","m_date"],
                                   right_on = ["gvkey","date"], how = "left").set_index(["gvkey","date"])
        del input_data["m"], input_data["w"]["m_date"]
        input_data = pd.merge(input_data["w"].reset_index(),input_data["q"], left_on = ["gvkey","q_date"],
                              right_on = ["gvkey","date"], how = "left").set_index(["gvkey","date"])
        del input_data["q_date"]
        
        #### save files for output, mcap, ff3, ff5+m
        ## save output data
        output_data[columns_output].to_hdf(phd3_PATH.format(variant=f"{variant}_benchmark")+ \
                           f"ffy.h5",key="data")
        
        ## save mcap data
        input_data[["mcap"]].to_hdf(phd3_PATH.format(variant=f"{variant}_benchmark")+ \
                                    "mcap.h5",key="data")
        
        ## save ff3
        input_data[columns_ff3].to_hdf(phd3_PATH.format(variant=f"{variant}_benchmark")+ \
                                       "ff3.h5",key="data")
        
        ## save ff5
        input_data[columns_ff5M].to_hdf(phd3_PATH.format(variant=f"{variant}_benchmark")+ \
                                        "ff5M.h5",key="data")
        
        
        
        
        
        