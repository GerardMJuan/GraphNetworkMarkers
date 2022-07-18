"""
Create file to generate intra and inter correlations
For SC and FC metrics.

SC and FC with the same preprocessing as in the extracted values.
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
from itertools import permutations

def par_extract_corr(row):
    """
    Extract the correlations for both matrices
    
    """
    df_G = pd.DataFrame()

    subID = row.SubjID
    type_dir = row.CENTER

    print(subID + " " + type_dir)

    #subj_dir_id = subj_dir_id = f'C:/Users/gerar/Documents/output_fmri_dti/{type_dir}_{subID}'
    # subj_dir_id = f'/mnt/Bessel/Gproj/Gerard_DATA/MAGNIMS2021/output_fmri_dti/{type_dir}_{subID}'
    #subj_dir_id = f'C:/Users/gerar/Documents/output_CONN/{type_dir}_{subID}'
    #if not os.path.isfile(subj_dir_id+'/results/r_matrix.csv'): subj_dir_id = f'C:/Users/gerar/Documents/output_fmri_dti/{type_dir}_{subID}'
    
    subj_dir_id = f'/mnt/Bessel/Gproj/Gerard_DATA/MAGNIMS2021/output_CONN/{type_dir}_{subID}'
    if not os.path.isfile(subj_dir_id+'/results/r_matrix.csv'): subj_dir_id = f'/mnt/Bessel/Gproj/Gerard_DATA/MAGNIMS2021/output_fmri_dti/{type_dir}_{subID}'
    else: print("yes conn")
    # patillada pero gl
    idx_G = len(df_G) - 1
    # idx_nodes = len(df_nodes) - 1

    df_G.at[idx_G, "SubjID"] = subID
    df_G.at[idx_G, "CENTER"] = type_dir

    ### FC
    FC_path = f"{subj_dir_id}/results/r_matrix.csv"
    FC = np.loadtxt(FC_path, delimiter=',')
    # ONLY CORTICAL
    FC_left = FC[np.ix_(np.r_[14:45], np.r_[14:45])]
    FC_right = FC[np.ix_(np.r_[45:76], np.r_[45:76])]
    FC_inter = FC[np.ix_(np.r_[14:45], np.r_[45:76])]

    # CORTICAL AND SUBCORTICAL
    #FC_left = FC[np.ix_(np.r_[0:7,14:45], np.r_[0:7,14:45])]
    #FC_right = FC[np.ix_(np.r_[7:14,45:76], np.r_[7:14,45:76])]
    #FC_inter = FC[np.ix_(np.r_[0:7,14:45], np.r_[7:14,45:76])]

    # Compute mean correlations
    # mean of all the values of the superior triangle, without the diagonal
    FC_Corr_intra = np.mean(FC_left[np.triu_indices(FC_left.shape[0], 1)])
    #ONLY homotopic connections
    FC_Corr_inter = np.mean([FC_inter[i,i] for i in range(len(FC_inter))])
    df_G["FC_Corr_inter"] = FC_Corr_inter
    df_G["FC_Corr_intra"] = FC_Corr_intra

    # load SC and tract length matrices
    SC_path = f"{subj_dir_id}/results/{type_dir}_{subID}_SC_weights.txt"
    len_path = f"{subj_dir_id}/results/{type_dir}_{subID}_SC_distances.txt"

    SC = np.loadtxt(open(SC_path, "rb"), delimiter=" ", skiprows=0)
    SC_len = np.loadtxt(open(len_path, "rb"), delimiter=" ", skiprows=0)

    #### NORMALIZE BY LEN
    SC = SC / SC_len
    SC = np.nan_to_num(SC, nan=0, posinf=0, neginf=0)

    # ONLY CORTICAL
    SC_left = SC[np.ix_(np.r_[14:45], np.r_[14:45])]
    SC_right = SC[np.ix_(np.r_[45:76], np.r_[45:76])]
    SC_inter = SC[np.ix_(np.r_[14:45], np.r_[45:76])]

    # CORTICAL AND SUBCORTICAL
    #SC_left = SC[np.ix_(np.r_[0:7,14:45], np.r_[0:7,14:45])]
    #SC_right = SC[np.ix_(np.r_[7:14,45:76], np.r_[7:14,45:76])]
    #SC_inter = SC[np.ix_(np.r_[0:7,14:45], np.r_[7:14,45:76])]

    # Compute mean connections
    # SC_Corr_intra = (np.mean(SC_left) + np.mean(SC_right)) / 2
    SC_Corr_intra = np.mean(SC_left[np.triu_indices(SC_left.shape[0], 1)])

    # SC_Corr_inter = np.mean(SC_inter)
    #ONLY homotopic connections
    SC_Corr_inter = np.mean([SC_inter[i,i] for i in range(len(SC_inter))])

    df_G["SC_Corr_inter"] = SC_Corr_inter
    df_G["SC_Corr_intra"] = SC_Corr_intra


    return df_G

# python intra_inter_corr.py --total_csv /home/extop/GERARD/DATA/MAGNIMS2021/data_total.csv --pip_csv /home/extop/GERARD/DATA/MAGNIMS2021/pipeline.csv --out_csv_prefix  /home/extop/GERARD/DATA/MAGNIMS2021/graph_values/graph --njobs 1
# python intra_inter_corr.py --total_csv C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/data_total.csv --pip_csv C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/pipeline.csv --out_csv_prefix  C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/graph_values/graph --njobs 1
@click.command(help="Run over the existing subjects, load the networks and extract their values.")
@click.option("--total_csv", required=True, type=click.STRING, help="csv with the base information for every subject")
@click.option("--pip_csv", required=True, type=click.STRING, help="csv with the current pipeline information for every subject")
@click.option("--out_csv_prefix", required=True, type=click.STRING, help="Output csv prefix. Will output various csv files")
@click.option("--njobs", required=True, type=click.STRING, help="number of jobs")
def intra_inter_corr(total_csv, pip_csv, out_csv_prefix, njobs):
    """
    Compute T1 values
    """

    # iterate over the subjects
    df_total = pd.read_csv(total_csv)
    df_pipeline = pd.read_csv(pip_csv)

    # will save everything in dictionaries to save it later to df. Columns are labels
    df_G = pd.DataFrame()

    njobs = int(njobs)

    # at least dt status, so that we have processed lesions volumes
    results = [par_extract_corr(row) for row in df_total.itertuples()\
                if df_pipeline[(df_pipeline.id==row.SubjID) & (df_pipeline.CENTER==row.CENTER)]["agg_SC"].bool() &
                   df_pipeline[(df_pipeline.id==row.SubjID) & (df_pipeline.CENTER==row.CENTER)]["fMRI"].bool()]

    list_of_G = [G for G in results]
    # list_of_nodes = [nodes for (_, nodes) in results]
    
    df_G = pd.concat(list_of_G)
    # df_nodes = pd.concat(list_of_nodes)

    # save to csv
    df_G.to_csv(f'{out_csv_prefix}_SC_FC_corr.csv')
    # df_nodes.to_csv(f'{out_csv_prefix}_nodes_SC.csv')


if __name__ == "__main__":
    # those parameters have to be entered from outside
    intra_inter_corr()