"""
Extract SC values

Update this script as new values are extracted. The markers extracted are:

- Weighted clustering coefficient
- Average connectivity
- Global efficiency (SC lengths)
- global centrality
- MSP
- MST

This script follow similar structure to the one on creating MR values, from MSclerosis-TVB
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
from community import best_partition, modularity
from itertools import combinations, product
import networkx as nx
from itertools import permutations

# python extract_SC_values.py --total_csv /home/extop/GERARD/DATA/MAGNIMS2021/data_total.csv --pip_csv /home/extop/GERARD/DATA/MAGNIMS2021/pipeline.csv --out_csv_prefix  /home/extop/GERARD/DATA/MAGNIMS2021/graph_values/graph --njobs 1 /home/extop/GERARD/DATA/MAGNIMS2021

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
    # will save everything in dictionaries to save it later to df. Columns are labels
    df_G = pd.DataFrame()
    # df_nodes = pd.DataFrame()

    subID = row.SubjID
    type_dir = row.CENTER

    print(subID + " " + type_dir)

    subj_dir_id = f'{subj_dir}/{type_dir}_Post/{subID}'

    """
    CLINIC OG PART
    COMMENT IF NEEDED
    """
    # for the clinic part   
    # clinic_og_path = f"/mnt/nascarm01/data/Projectes/MAGNIMS2021/CLINIC/{subID}"
    # SC_path = f"{clinic_og_path}/r{subID}_SC_raw.csv"
    # SC_path_raw = f"rFIS_{subID}_SC_raw.csv"

    # names = pd.read_csv(f"{subj_dir_id}/results/centres.txt", header=None, usecols=[0], sep=' ')
    # names = [x for [x] in names.values]

    # patillada pero gl
    idx_G = len(df_G) - 1
    # idx_nodes = len(df_nodes) - 1

    df_G.at[idx_G, "SubjID"] = subID
    df_G.at[idx_G, "CENTER"] = type_dir
    # df_nodes.at[idx_nodes, "SubjID"] = subID
    # df_nodes.at[idx_nodes, "CENTER"] = type_dir

    # load SC and tract length matrices
    SC_path = f"{subj_dir_id}/dt_proc/connectome_weights.csv"
    len_path = f"{subj_dir_id}/dt_proc/connectome_lengths.csv"

    SC = np.loadtxt(open(SC_path, "rb"), delimiter=",", skiprows=0)
    SC_len = np.loadtxt(open(len_path, "rb"), delimiter=",", skiprows=0)

    #### NORMALIZE BY LEN
    SC = SC / SC_len
    SC = np.nan_to_num(SC, nan=0, posinf=0, neginf=0)
    Comm_ratio = np.sum(SC[np.ix_(np.r_[14:45], np.r_[45:76])]) / np.sum(SC)
    # Comm_ratio = np.sum(SC[np.ix_(np.r_[0:7,14:45], np.r_[7:14,45:76])]) / np.sum(SC)
    # Comm_ratio = np.sum(SC[np.ix_(np.r_[0:38], np.r_[38:76])])*2 / np.sum(SC)
    
    df_G.at[idx_G, f'Comm_ratio'] = Comm_ratio

    # FC
    """
    FC_path = f"{subj_dir_id}/fmri_proc_dti/r_matrix.csv"
    FC = np.loadtxt(FC_path, delimiter=',')
    # remove all negative weights
    FC[FC < 0] = 0.0

    FC = FC / np.amax(FC)    
    
    FC_left = FC[np.ix_(np.r_[0:7,14:45], np.r_[0:7,14:45])]
    FC_right = FC[np.ix_(np.r_[7:14,45:76], np.r_[7:14,45:76])]
    FC_inter = FC[np.ix_(np.r_[0:7,14:45], np.r_[7:14,45:76])]
    # FC_inter[np.ix_(np.r_[0:7,14:45], np.r_[7:14,45:76])] = 0.0

    FC_left_nx = nx.from_numpy_matrix(FC_left)
    FC_right_nx = nx.from_numpy_matrix(FC_right)
    FC_inter_nx = nx.from_numpy_matrix(FC_inter)
    """
    
    # convert zero values to inf

    # load them in networkx
    # BASE
    # SC_left = SC[np.ix_(np.r_[0:7,14:45], np.r_[0:7,14:45])]
    # SC_right = SC[np.ix_(np.r_[7:14,45:76], np.r_[7:14,45:76])]
    # SC_inter = SC.copy()
    
    # CLINIC
    # SC_right = SC[np.ix_(np.r_[0:38], np.r_[0:38])]
    # SC_left = SC[np.ix_(np.r_[38:76], np.r_[38:76])]
    # SC_inter = SC.copy()

    # ONLY CORTICAL
    SC_left = SC[np.ix_(np.r_[14:45], np.r_[14:45])]
    SC_right = SC[np.ix_(np.r_[45:76], np.r_[45:76])]
    SC_inter = SC[np.ix_(np.r_[14:76], np.r_[14:76])]

    # SC_inter[np.ix_(np.r_[0:7,14:45], np.r_[7:14,45:76])] = 0.0
    ### Save to disk the matrices to observe them

    # SC_left = (SC_left / np.sum(SC_left)) / np.amax(SC_left / np.sum(SC_left))
    # SC_right = (SC_right / np.sum(SC_right)) / np.amax(SC_right / np.sum(SC_right))
    # SC_inter = (SC_inter / np.sum(SC_inter)) / np.amax(SC_inter / np.sum(SC_inter))

    SC_mask_l = SC_left != 0
    SC_mask_r = SC_right != 0
    SC_mask_i = SC_inter != 0

    # SC_left[SC_mask_l] = SC_left[SC_mask_l] / np.amax(SC_left)
    # SC_right[SC_mask_r] = SC_right[SC_mask_r] / np.amax(SC_right)
    # SC_inter[SC_mask_i] = SC_inter[SC_mask_i] / np.amax(SC_inter)

    """
    plt.imshow(SC_left)
    plt.colorbar()
    plt.savefig("SC_left.png")
    plt.close()
    plt.imshow(SC_right)
    plt.colorbar()
    plt.savefig("SC_right.png")
    plt.close()
    plt.imshow(SC)
    plt.colorbar()
    plt.savefig("SC.png")
    plt.close()
    """

    SC_left[SC_mask_l] = 1 / SC_left[SC_mask_l]
    SC_right[SC_mask_r] = 1 / SC_right[SC_mask_r]
    SC_inter[SC_mask_i] = 1 / SC_inter[SC_mask_i]

    SC_left_nx = nx.from_numpy_matrix(SC_left)
    SC_right_nx = nx.from_numpy_matrix(SC_right)
    SC_inter_nx = nx.from_numpy_matrix(SC_inter)

    # Create pairs of nodes that need to be added. Three lists:
    # 1) pairs of nodes in hemisphere L
    # 2) pairs of nodes in hemisphere R
    # 3) pairs of nodes across hemispheres
    # as we have no directed graphs, use only direct paths
    pairs_L = [x for x in combinations([x for x in range(len(SC_left))], 2)]
    pairs_R = [x for x in combinations([x for x in range(len(SC_left))], 2)]
    
    # Base
    # pairs_inter = [x for x in product(np.r_[0:7,14:45], np.r_[7:14,45:76])]

    # ONLY CORTICAL
    pairs_inter = [x for x in product(np.r_[14:45], np.r_[45:76])]

    # iterate over
    list_of_graphs = [SC_left_nx, SC_right_nx, SC_inter_nx]#, FC_left_nx, FC_right_nx, FC_inter_nx] #, Tract_ntx] #, FC_ntx]
    list_of_names = ["SC_L", "SC_R", "SC_inter"]#, "FC_R", "FC_inter"] #, "Tract"] #, "FC"]
    list_of_pairs = [permutations(SC_left_nx, 2), permutations(SC_right_nx, 2), permutations(SC_inter_nx, 2)]#, pairs_L, pairs_R, pairs_inter]

    # REMOVE EDGES WITH WEIGHT 0
    # for G in list_of_graphs:
    #     long_edges = list(filter(lambda e: e[2] == 0, (e for e in G.edges.data('weight'))))
    #     le_ids = list(e[:2] for e in long_edges)
        # remove filtered edges from graph G
    #     G.remove_edges_from(le_ids)

    # iterate over the existing graphs, later we could add FC
    for (graph, name, pairs) in zip(list_of_graphs, list_of_names, list_of_pairs):
        # Compute the various values
        # and save them depending on if its by node or by general graph
        # save each value
        # avg_clus = nx.algorithms.cluster.average_clustering(graph, weight='weight')
        # df_G.at[idx_G, f'{name}_avg_clus'] = avg_clus

        # avg_clus = nx.algorithms.generic.average_shortest_path_length(graph, weight='weight')
        # df_G.at[idx_G, f'{name}_avg_spl'] = avg_clus

        """
        ## Weighted clustering coefficiency
        ### manual
        spl = dict(nx.algorithms.shortest_paths.generic.shortest_path_length(graph, weight='weight'))
        pairs_spl_arr = []
        npairs = 0
        for (x, y) in pairs:
            # if we don't find it, that means that that connection doesnt exist.
            try:
                pairs_spl_arr.append(1/spl[x][y])
            except KeyError:
                pass
            npairs += 1
        spl_avg_pairs = np.sum(pairs_spl_arr) / npairs
        df_G.at[idx_G, f'{name}_avg_spl'] = spl_avg_pairs
        """
        df_G.at[idx_G, f'{name}_avg_spl'] = global_efficiency_weighted(graph, pairs)


        # for inter, compute it manually

        # i = 0 # i is an indexer, to iterate the nodes of the network and select the names
        # for n in names:
        #     df_nodes.at[idx_nodes, f'{name}_{n}_clustering'] = avg_clus_nodes[i]
        #     i += 1

        ## Average connectivity
        # NOTE: should be do some kind of thresholding?
        # avg_conn = nx.algorithms.connectivity.connectivity.average_node_connectivity(graph)
        # df_G.at[idx_G, f'{name}_avg_conn'] = avg_conn

        ## Global efficiency
        # average of the inverse shortest path length between all nodes in the network
        # spl = dict(nx.algorithms.shortest_paths.generic.shortest_path_length(graph, weight='weight'))
        # df_G.at[idx_G, f'{name}_avg_spl'] = spl_avg
        

        """
        L_spl_arr = []
        R_spl_arr = []
        pairs_spl_arr = []

        L_glob_arr = []
        R_glob_arr = []
        pairs_glob_arr = []

        for (x, y) in pairs_L:
            L_spl_arr.append(spl[x][y])
            L_glob_arr.append(1/spl[x][y])
        for (x, y) in pairs_R:
            R_spl_arr.append(spl[x][y])
            R_glob_arr.append(1/spl[x][y])
        for (x, y) in pairs_inter:
            pairs_spl_arr.append(spl[x][y])
            pairs_glob_arr.append(1/spl[x][y])

        global_efficiency_L = np.sum(L_glob_arr) / len(L_glob_arr)
        spl_avg_L = np.sum(L_spl_arr) / len(L_spl_arr)

        global_efficiency_R = np.sum(R_glob_arr) / len(R_glob_arr)
        spl_avg_R = np.sum(R_spl_arr) / len(R_spl_arr)

        global_efficiency_pairs = np.sum(pairs_glob_arr) / len(pairs_glob_arr)
        spl_avg_pairs = np.sum(pairs_spl_arr) / len(pairs_spl_arr)

        # spl_avg = np.sum(L_spl_arr + R_spl_arr + pairs_spl_arr) / ( len(L_spl_arr) + len(R_spl_arr) + len(pairs_spl_arr) )

        df_G.at[idx_G, f'{name}_avg_conn_L'] = global_efficiency_L
        df_G.at[idx_G, f'{name}_avg_spl_L'] = spl_avg_L

        df_G.at[idx_G, f'{name}_avg_conn_R'] = global_efficiency_R
        df_G.at[idx_G, f'{name}_avg_spl_R'] = spl_avg_R
        
        df_G.at[idx_G, f'{name}_avg_conn_pairs'] = global_efficiency_pairs
        df_G.at[idx_G, f'{name}_avg_spl_pairs'] = spl_avg_pairs
        """

        ## Centrality
        ## do we need 
        # centrality = nx.algorithms.centrality.degree_centrality(graph)
        # i = 0 # i is an indexer, to iterate the nodes of the network and select the names
        # for n in names:
        #     df_nodes.at[idx_nodes, f'{name}_{n}_centrality'] = centrality[i]
        #     i += 1

        ## Eigenvector Centrality
        # centrality = nx.algorithms.centrality.eigenvector_centrality(graph, weight='weight', max_iter=500)
        # i = 0 # i is an indexer, to iterate the nodes of the network and select the names
        # for n in names:
        #     df_nodes.at[idx_nodes, f'{name}_{n}_eigen_centrality'] = centrality[i]
        #     i += 1

        ## Modularity:
        # Find best partition and compute modularity of the graph
        # partition = best_partition(graph, weight='weight')
        # mod = modularity(partition, graph, weight='weight')
        # df_G.at[idx_G, f'{name}_modularity'] = mod

        ## Small world omega
        # we already have clustering and short path lengths
        # generate the random and lattice graphs, and compute the 
        
        # average shortest path length of an equivalent random graph
        # rd_graph = nx.algorithms.smallworld.random_reference(graph)
        # L_r = nx.algorithms.average_shortest_path_length(rd_graph, weight='weight')

        # Cl is the average clustering coefficient of an equivalent lattice graph.
        # lt_graph = nx.algorithms.smallworld.lattice_reference(graph)
        # C_l = nx.algorithms.cluster.average_clustering(lt_graph, weight='weight')

        # compute omega small world
        # omega_sw = L_r/spl_avg - avg_clus/C_l
        # df_G.at[idx_G, f'{name}_smallworld'] = omega_sw
    
    # return (df_G, df_nodes)
    return df_G


@click.command(help="Run over the existing subjects, load the networks and extract their values.")
@click.option("--total_csv", required=True, type=click.STRING, help="csv with the base information for every subject")
@click.option("--pip_csv", required=True, type=click.STRING, help="csv with the current pipeline information for every subject")
@click.option("--out_csv_prefix", required=True, type=click.STRING, help="Output csv prefix. Will output various csv files")
@click.option("--njobs", required=True, type=click.STRING, help="number of jobs")
@click.argument("subj_dir")
def compute_SC_values(subj_dir, total_csv, pip_csv, out_csv_prefix, njobs):
    """
    Compute T1 values
    """

    # iterate over the subjects
    df_total = pd.read_csv(total_csv)
    df_pipeline = pd.read_csv(pip_csv)

    ## CLINIC
    # df_total = df_total[df_total.CENTER=="CLINIC"]
    # df_pipeline = df_pipeline[df_pipeline.CENTER=="CLINIC"]

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
    df_G.to_csv(f'{out_csv_prefix}_G_SC.csv')
    # df_nodes.to_csv(f'{out_csv_prefix}_nodes_SC.csv')


if __name__ == "__main__":
    # those parameters have to be entered from outside
    compute_SC_values()
