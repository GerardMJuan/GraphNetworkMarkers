"""
Extract FC values
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
# from community import best_partition, modularity
from itertools import combinations, product
import networkx as nx
from itertools import permutations

# python extract_FC_values.py --total_csv /home/extop/GERARD/DATA/MAGNIMS2021/data_total.csv --pip_csv /home/extop/GERARD/DATA/MAGNIMS2021/pipeline.csv --out_csv_prefix  /home/extop/GERARD/DATA/MAGNIMS2021/graph_values/graph --njobs 1 /home/extop/GERARD/DATA/MAGNIMS2021
# python extract_FC_values.py --total_csv C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/data_total.csv --pip_csv C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/pipeline.csv --out_csv_prefix  C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/graph_values/graph --njobs 1 C:/Users/gerar/Documents/output_CONN

def global_efficiency_weighted(G, pairs):
    """
    Compute global efficiency.

    From: https://stackoverflow.com/questions/56554132/how-can-i-calculate-global-efficiency-more-efficiently
    """
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight = 'weight'))
        full_sum = [shortest_paths[u][v] for u, v in pairs if v in shortest_paths[u] and shortest_paths[u][v] != 0]
        g_eff = np.mean(full_sum)
    else:
        g_eff = 0
    return g_eff


def par_extract_values(row, subj_dir):
    """
    Auxiliar function to parallelize the extraction of values
    
    FC should also be here
    """
    df_G = pd.DataFrame()

    subID = row.SubjID
    type_dir = row.CENTER

    print(subID + " " + type_dir)

    subj_dir_id = f'{subj_dir}/{type_dir}_Post/{subID}'
    if not os.path.isfile(subj_dir_id+'/results/r_matrix.csv'): subj_dir_id = f'C:/Users/gerar/Documents/output_fmri_dti/{type_dir}_{subID}'

    # patillada pero gl
    idx_G = len(df_G) - 1
    # idx_nodes = len(df_nodes) - 1

    df_G.at[idx_G, "SubjID"] = subID
    df_G.at[idx_G, "CENTER"] = type_dir

    # FC
    FC_path = f"{subj_dir_id}/results/r_matrix.csv"
    FC = np.loadtxt(FC_path, delimiter=',')

    # threshold and create graph 
    #tenth_percentile = np.percentile(FC, 90)
    #print(tenth_percentile)
    FC = np.where(FC>0.5, 1, 0)
    
    # ONLY CORTICAL
    FC_left = FC[np.ix_(np.r_[14:45], np.r_[14:45])]
    FC_right = FC[np.ix_(np.r_[45:76], np.r_[45:76])]
    FC_inter = FC[np.ix_(np.r_[14:76], np.r_[14:76])]

    FC_left_nx = nx.from_numpy_matrix(FC_left)
    FC_right_nx = nx.from_numpy_matrix(FC_right)
    FC_inter_nx = nx.from_numpy_matrix(FC_inter)
    
    # iterate over
    list_of_graphs = [FC_left_nx, FC_right_nx, FC_inter_nx]#, FC_left_nx, FC_right_nx, FC_inter_nx] #, Tract_ntx] #, FC_ntx]
    list_of_names = ["FC_L", "FC_R", "FC_inter"]#, "FC_R", "FC_inter"] #, "Tract"] #, "FC"]
    list_of_pairs = [permutations(FC_left_nx, 2), permutations(FC_right_nx, 2), permutations(FC_inter_nx, 2)]#, pairs_L, pairs_R, pairs_inter]

    # iterate over the existing graphs, later we could add FC
    for (graph, name, pairs) in zip(list_of_graphs, list_of_names, list_of_pairs):
        largest_cc = max(nx.connected_components(graph), key=len)
        G = graph.subgraph(largest_cc).copy()
                
        # Avg efficiency
        df_G.at[idx_G, f'{name}_avg_spl'] = nx.global_efficiency(G)

        # mean shortest path length
        df_G.at[idx_G, f'{name}_efficiency'] = nx.average_shortest_path_length(G)

    # return (df_G, df_nodes)
    return df_G


@click.command(help="Run over the existing subjects, load the networks and extract their values.")
@click.option("--total_csv", required=True, type=click.STRING, help="csv with the base information for every subject")
@click.option("--pip_csv", required=True, type=click.STRING, help="csv with the current pipeline information for every subject")
@click.option("--out_csv_prefix", required=True, type=click.STRING, help="Output csv prefix. Will output various csv files")
@click.option("--njobs", required=True, type=click.STRING, help="number of jobs")
@click.argument("subj_dir")
def compute_FC_values(subj_dir, total_csv, pip_csv, out_csv_prefix, njobs):
    """
    Compute T1 values
    """

    # iterate over the subjects
    df_total = pd.read_csv(total_csv)
    df_pipeline = pd.read_csv(pip_csv)

    ## CLINIC

    # will save everything in dictionaries to save it later to df. Columns are labels
    df_G = pd.DataFrame()
    # df_nodes = pd.DataFrame()

    njobs = int(njobs)

    # at least dt status, so that we have processed lesions volumes
    # HACK: SELECT ONLY 5 SUBJECTS, TO TEST
    results = Parallel(n_jobs=njobs, backend="threading")(delayed(par_extract_values)(row, subj_dir) for row in df_total.itertuples()\
                                                                                      if df_pipeline[(df_pipeline.id==row.SubjID) & (df_pipeline.CENTER==row.CENTER)]["agg_SC"].bool() &
                                                                                         df_pipeline[(df_pipeline.id==row.SubjID) & (df_pipeline.CENTER==row.CENTER)]["fMRI"].bool())

    list_of_G = [G for G in results]
    # list_of_nodes = [nodes for (_, nodes) in results]
    
    df_G = pd.concat(list_of_G)
    # df_nodes = pd.concat(list_of_nodes)

    # save to csv
    df_G.to_csv(f'{out_csv_prefix}_G_FC.csv')
    # df_nodes.to_csv(f'{out_csv_prefix}_nodes_SC.csv')


if __name__ == "__main__":
    # those parameters have to be entered from outside
    compute_FC_values()
