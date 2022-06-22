"""
Extract FC values

Update this script as new values are extracted. The markers extracted are:

- Weighted clustering coefficient
- Average connectivity
- Global efficiency (FC lengths)
- global centrality
- MSP
- MST

This script follow similar structure to the one on creating MR values, from MSclerosis-TVB

Remember that the FC has negative values. So we need to decide if we use thresholding or what do we use
"""
import numpy as np
import os
import click
import subprocess
import numpy as np
import datetime
import pandas as pd
import networkx as nx
from joblib import Parallel, delayed
from community import best_partition, modularity

# python extract_FC_values.py --total_csv /home/extop/GERARD/DATA/MAGNIMS2021/data_total.csv --pip_csv /home/extop/GERARD/DATA/MAGNIMS2021/pipeline.csv --out_csv_prefix  /home/extop/GERARD/DATA/MAGNIMS2021/graph_values/graph --njobs 1 /home/extop/GERARD/DATA/MAGNIMS2021

def par_extract_values(row, subj_dir):
    """
    Auxiliar function to parallelize the extraction of values
    
    FC should also be here
    """
    # will save everything in dictionaries to save it later to df. Columns are labels
    df_G = pd.DataFrame()
    df_nodes = pd.DataFrame()

    subID = row.SubjID
    type_dir = row.CENTER

    print(subID + " " + type_dir)

    subj_dir_id = f'{subj_dir}/{type_dir}_Post/{subID}'

    names = pd.read_csv(f"{subj_dir_id}/results/centres.txt", header=None, usecols=[0], sep=' ')
    names = names.values

    df_G["SubjID"] = subID
    df_G["CENTER"] = type_dir
    df_nodes["SubjID"] = subID
    df_nodes["CENTER"] = type_dir

    # patillada pero gl
    idx_G = len(df_G) - 1
    idx_nodes = len(df_nodes) - 1


    # FC
    FC_path = f"{subj_dir_id}/results/r_matrix.csv"
    FC = np.loadtxt(FC_path, delimiter=',')


    # sum 1 and divide per 2, akin to Tewarie et al 2015
    FC = (FC + 1.0) / 2.0
    FC_ntx = nx.from_numpy_matrix(FC)

    # iterate over
    list_of_graphs = [FC_ntx]
    list_of_names = ["FC"]

    # iterate over the existing graphs, later we could add FC
    for (graph, name) in zip(list_of_graphs, list_of_names):
        # Compute the various values
        # and save them depending on if its by node or by general graph
        # save each value

        ## Weighted clustering coefficiency
        avg_clus = nx.algorithms.cluster.average_clustering(graph, weight='weight')
        avg_clus_nodes = nx.algorithms.cluster.clustering(graph, weight='weight')

        df_G.at[idx_G, f'{name}_avg_clus'] = avg_clus

        i = 0 # i is an indexer, to iterate the nodes of the network and select the names
        for n in names:
            df_nodes.at[idx_nodes, f'{name}_{n}_clustering'] = avg_clus_nodes[i]
            i += 1

        ## Average connectivity
        # NOTE: should be do some kind of thresholding?
        avg_conn = nx.algorithms.connectivity.connectivity.average_node_connectivity(graph)
        df_G.at[idx_G, f'{name}_avg_conn'] = avg_conn

        ## Global efficiency
        # average of the inverse shortest path length between all nodes in the network
        # ERROR in FC FOR NEGATIVE WEIGHTS. WHAT TO DO? 
        spl = nx.algorithms.shortest_paths.generic.shortest_path_length(graph, weight='weight')

        # sanity check: those two values should be the same

        global_efficiency = 0
        spl_avg = 0
        i = 0 # iterator to check which region are we on
        N = 0 # cumulative sum to divide later to obtain mean
        for (_, dict) in spl: # for each node
            for (k, x) in dict.items():
                if k <= i: continue
                global_efficiency += 1/x
                spl_avg += x
                N += 1
            i += 1

        global_efficiency = global_efficiency / N
        spl_avg = spl_avg / N
        
        df_G.at[idx_G, f'{name}_avg_conn'] = global_efficiency
        df_G.at[idx_G, f'{name}_avg_spl'] = spl_avg

        ## Centrality
        ## do we need 
        centrality = nx.algorithms.centrality.degree_centrality(graph)
        i = 0 # i is an indexer, to iterate the nodes of the network and select the names
        for n in names:
            df_nodes.at[idx_nodes, f'{name}_{n}_centrality'] = centrality[i]
            i += 1

        ## Eigenvector Centrality
        centrality = nx.algorithms.centrality.eigenvector_centrality(graph, weight='weight', max_iter=500)
        i = 0 # i is an indexer, to iterate the nodes of the network and select the names
        for n in names:
            df_nodes.at[idx_nodes, f'{name}_{n}_eigen_centrality'] = centrality[i]
            i += 1

        ## Modularity:
        # Find best partition and compute modularity of the graph
        partition = best_partition(graph, weight='weight')
        mod = modularity(partition, graph, weight='weight')
        df_G.at[idx_G, f'{name}_modularity'] = mod

        ## Small world omega
        # we already have clustering and short path lengths
        # generate the random and lattice graphs, and compute the 
        
        #   CANT COMPUTE IN FC WITHOUT CHANGES
        # average shortest path length of an equivalent random graph
        rd_graph = nx.algorithms.smallworld.random_reference(graph)
        L_r = nx.algorithms.average_shortest_path_length(rd_graph, weight='weight')

        # Cl is the average clustering coefficient of an equivalent lattice graph.
        lt_graph = nx.algorithms.smallworld.lattice_reference(graph)
        C_l = nx.algorithms.cluster.average_clustering(lt_graph, weight='weight')

        # compute omega small world
        omega_sw = L_r/spl_avg - avg_clus/C_l
        df_G.at[idx_G, f'{name}_smallworld'] = omega_sw
    return (df_G, df_nodes)



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

    # will save everything in dictionaries to save it later to df. Columns are labels
    df_G = pd.DataFrame()
    df_nodes = pd.DataFrame()

    njobs = int(njobs)

    # at least dt status, so that we have processed lesions volumes
    # HACK: SELECT ONLY 5 SUBJECTS, TO TEST
    results = Parallel(n_jobs=njobs, backend="threading")(delayed(par_extract_values)(row, subj_dir) for row in df_total.itertuples()\
                                                                                      if df_pipeline[(df_pipeline.id==row.SubjID) & (df_pipeline.CENTER==row.CENTER)]["toTVB"].bool())

    list_of_G = [G for (G, _) in results]
    list_of_nodes = [nodes for (_, nodes) in results]

    df_G = pd.concat(list_of_G)
    df_nodes = pd.concat(list_of_nodes)

    # save to csv
    df_G.to_csv(f'{out_csv_prefix}_G_FC.csv')
    df_nodes.to_csv(f'{out_csv_prefix}_nodes_FC.csv')


if __name__ == "__main__":
    # those parameters have to be entered from outside
    compute_FC_values()
