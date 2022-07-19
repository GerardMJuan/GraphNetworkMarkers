"""
Secondary script to calculate correlation of short/long pathways in the SC and FC

Akin to paper:
1. Kulik SD, Nauta IM, Tewarie P, et al. 
Structure-function coupling as a correlate and potential biomarker of cognitive impairment in multiple sclerosis. 
Netw Neurosci. 2022;6(2):339-356. doi:10.1162/netn_a_00226

"""

import numpy as np
import os
import click
import subprocess
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from itertools import combinations, product
import networkx as nx
from itertools import permutations

def extract_quartiles_short_long(subj_dir, df_healthy, cortical):
    """
    Load all the healthy tracts matrices in df_healthy,
    put in into a matrix, and return first and last quartile
    """
    #Load FC
    distance_list = np.array([])

    for row in df_healthy.itertuples():
        subID = row.SubjID
        type_dir = row.CENTER

        subj_dir_id = f'{subj_dir}/{type_dir}_{subID}'
        len_path = f"{subj_dir_id}/results/{type_dir}_{subID}_SC_distances.txt"
        SC_len = np.loadtxt(open(len_path, "rb"), skiprows=0)

        # ONLY CORTICAL
        if cortical:
            SC_len = SC_len[np.ix_(np.r_[14:76], np.r_[14:76])]


        distance_list = np.concatenate((distance_list, SC_len), axis=None)
    distance_list = distance_list[distance_list != 0]
    q1 = np.percentile(distance_list, 0.25)
    q4 = np.percentile(distance_list, 0.75)
    return q1, q4

def par_extract_values(row, subj_dir, cortical, quartiles_dict):
    """
    Auxiliar function to parallelize the extraction of values
    
    FC should also be here
    """
    df_G = pd.DataFrame()

    subID = row.SubjID
    type_dir = row.CENTER

    print(subID + " " + type_dir)

    subj_dir_id = f'{subj_dir}/{type_dir}_{subID}'

    #Load FC
    try:
        # load SC and tract length matrices
        SC_path = f"{subj_dir_id}/results/{type_dir}_{subID}_SC_weights.txt"
        len_path = f"{subj_dir_id}/results/{type_dir}_{subID}_SC_distances.txt"

        SC = np.loadtxt(open(SC_path, "rb"), skiprows=0)
        SC_len = np.loadtxt(open(len_path, "rb"), skiprows=0)
        
        # FC
        FC_path = f"{subj_dir_id}/results/r_matrix.csv"
        FC = np.loadtxt(FC_path, delimiter=',')
        
        # Corr_ts
        timeseries = f"{subj_dir_id}/results/corrlabel_ts.txt"
        timeseries = np.loadtxt(timeseries)
        timeseries = timeseries.T
    except OSError:
        # això es suficient per fer les que estan completes?
        print(f'{subID}_{type_dir} not found!')
        return df_G

    # patillada pero gl
    idx_G = len(df_G) - 1

    df_G.at[idx_G, "SubjID"] = subID
    df_G.at[idx_G, "CENTER"] = type_dir

    #Separate nodes between short and long range
    q1 = quartiles_dict[type_dir][0]
    q4 = quartiles_dict[type_dir][1]

    # ONLY CORTICAL
    if cortical:
        SC = SC[np.ix_(np.r_[14:76], np.r_[14:76])]
        FC = FC[np.ix_(np.r_[14:76], np.r_[14:76])]

    # select the nodes
    q1_nodes = SC_len < q1
    q4_nodes = SC_len > q4

    #select the values from SC and FC in each quartile
    q1_SC = SC[q1_nodes]
    q4_SC = SC[q4_nodes]
    q1_FC = FC[q1_nodes]
    q4_FC = FC[q4_nodes]

    #Between subject correlaction
    df_G.at[idx_G, "SC_corr_whole"] = np.mean(SC[np.triu_indices(SC.shape[0], 1)])
    df_G.at[idx_G, "SC_corr_q1"] = np.mean(q1_SC[q1_SC != 0])
    df_G.at[idx_G, "SC_corr_q4"] = np.mean(q4_SC[q4_SC != 0])

    df_G.at[idx_G, "FC_corr_whole"] = np.mean(FC[np.triu_indices(FC.shape[0], 1)])
    df_G.at[idx_G, "FC_corr_q1"] = np.mean(q1_FC[q1_FC != 0])
    df_G.at[idx_G, "FC_corr_q4"] = np.mean(q4_FC[q4_FC != 0])

    # Within subject correlation
    df_G.at[idx_G, "short_FCSC"] = np.corrcoef(q1_FC.ravel(), q1_SC.ravel())[0,1]
    df_G.at[idx_G, "long_FCSC"] = np.corrcoef(q4_FC.ravel(), q4_SC.ravel())[0,1]

    return df_G

# python long_short_corr.py --total_csv C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/data_total.csv --pip_csv C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/pipeline.csv --out_csv_prefix  C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/graph_values/long_short_dti --njobs 1 C:/Users/gerar/Documents/output_dti_fmri

@click.command(help="Run over the existing subjects, load the networks and extract their values.")
@click.option("--total_csv", required=True, type=click.STRING, help="csv with the base information for every subject")
@click.option("--pip_csv", required=True, type=click.STRING, help="csv with the current pipeline information for every subject")
@click.option("--out_csv_prefix", required=True, type=click.STRING, help="Output csv prefix. Will output various csv files")
@click.option("--njobs", required=True, type=click.STRING, help="number of jobs")
@click.option('--cortical', is_flag=True, help="use only cortical values.")
@click.argument("subj_dir")
def compute_long_short_values(subj_dir, total_csv, pip_csv, out_csv_prefix, njobs, cortical):
    """
    Compute SC values
    """

    # iterate over the subjects
    df_total = pd.read_csv(total_csv)
    df_pipeline = pd.read_csv(pip_csv)

    njobs = int(njobs)

    quartiles_dict = {}

    #iterate over each value of CENTER on data_total
    for center in df_total.CENTER2.unique():
        if center == "LONDON2": continue
        df_center = df_total[df_total.CENTER2 == center]
        df_center = df_center[df_center.GROUP == "HC"]
        df_center = df_center[df_center.QC == "Y"]

        # get quartiles
        q1, q4 = extract_quartiles_short_long(subj_dir, df_center, cortical)

        #save quartiles with center as key
        quartiles_dict[center] = [q1, q4]

    # at least dt status, so that we have processed lesions volumes
    results = Parallel(n_jobs=njobs, backend="threading")(delayed(par_extract_values)(row, subj_dir, cortical, quartiles_dict) for row in df_total.itertuples()\
                                                                                      if df_pipeline[(df_pipeline.id==row.SubjID) & (df_pipeline.CENTER==row.CENTER)]["agg_SC"].bool() &
                                                                                         df_pipeline[(df_pipeline.id==row.SubjID) & (df_pipeline.CENTER==row.CENTER)]["fMRI"].bool())

    list_of_G = [G for G in results]
    
    df_G = pd.concat(list_of_G)

    # save to csv
    if cortical: 
        df_G.to_csv(f'{out_csv_prefix}_cort.csv')
    else:
        df_G.to_csv(f'{out_csv_prefix}.csv')


if __name__ == "__main__":
    # those parameters have to be entered from outside
    compute_long_short_values()
