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
from scipy.stats import entropy
from sklearn.metrics import auc
from scipy.signal import hilbert


def remove_mean(x, axis):
    """
    Remove mean from numpy array along axis
    """
    # Example for demean(x, 2) with x.shape == 2,3,4,5
    # m = x.mean(axis=2) collapses the 2'nd dimension making m and x incompatible
    # so we add it back m[:,:, np.newaxis, :]
    # Since the shape and axis are known only at runtime
    # Calculate the slicing dynamically
    idx = [slice(None)] * x.ndim
    idx[axis] = np.newaxis

    return x - x.mean(axis=axis)[tuple(idx)]  # / x.std(axis=axis)[idx]
    # return ( x - x.mean(axis=axis)[idx] ) # / x.std(axis=axis)[idx]


def kuromoto_metastability(time_series):
    """
    Compute metastability of the kuramoto index.
    Time series is a BOLD signal of shape nregions x timepoints.

    Papers of interest:
    Hellyer PJ, Shanahan M, Scott G, Wise RJ, Sharp DJ, Leech R.
    The control of global brain dynamics: opposing actions of frontoparietal control and default mode networks on attention.
    J Neurosci. 2014;34(2):451-461. doi:10.1523/JNEUROSCI.1853-13.2014

    Deco, G., Kringelbach, M.L., Jirsa, V.K. et al. The dynamics of resting fluctuations in the brain:
    metastability and its dynamical cortical core. Sci Rep 7, 3095 (2017). https://doi.org/10.1038/s41598-017-03073-5
    https://www.nature.com/articles/s41598-017-03073-5#Sec6
    """

    # input: demeaned BOLD signal of shape Nregions x timepoints
    hb = hilbert(time_series, axis=1)

    # polar torna una tuple que es ( modulus (abs), phase )
    # theta_sum = np.sum(np.exp(1j * (np.vectorize(cmath.polar)(hb)[1])), axis=0)
    theta = np.exp(1j * np.unwrap(np.angle(hb)))
    theta_sum = np.sum(theta, axis=0)

    # kuramoto = np.vectorize(cmath.polar)(theta_sum / time_series.shape[0])[0]
    kuramoto = np.abs(theta_sum) / time_series.shape[0]

    # ara que tinc el kuramoto, calcular metastability across time
    metastability = kuramoto.std()
    return metastability


def entropy_FC(FC):
    """
    Extract an entropy measure, as in Saenger et al.

    For each column of the FC, compute its entropy and calculate the mean.
    """
    entropy_list = []
    for i in range(FC.shape[0]):
        col_norm = (FC[i, :] - np.min(FC[i, :])) / (np.max(FC[i, :]) - np.min(FC[i, :]))
        e = entropy(col_norm, base=10)
        entropy_list.append(e)
    FC_entropy = np.mean(entropy_list)
    return FC_entropy


def integration_AUC(FC):
    """
    Extract an integration measure, which is the AUC of different thresholds over
    the network, akin to Adikhari 2017 et al.
    """
    th_ranges = np.linspace(0, 1, 50)
    sizes = []
    for th in th_ranges:
        # binarize with the th
        FC_th = np.where(FC > th, 1, 0)

        # create graph
        G = nx.from_numpy_matrix(FC_th)

        # find largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        largest_cc = G.number_of_nodes()
        sizes.append(largest_cc)

    # compute AUC
    AUC_value = auc(th_ranges, sizes)
    return AUC_value


def par_extract_values(row, subj_dir, th, cortical):
    """
    Auxiliar function to parallelize the extraction of values

    FC should also be here
    """
    df_G = pd.DataFrame()

    subID = row.SubjID
    type_dir = row.CENTER

    print(subID + " " + type_dir)
    subj_dir_id = f"{subj_dir}/{type_dir}_{subID}"

    # iterate over all existing FC in that directory
    # plus the average one
    # create auxiliar function that, for a given FC/timeseries, computes all the values,
    # saves it in a dict, and returns it

    try:
        # FC
        FC_path = f"{subj_dir_id}/results/r_matrix.csv"
        FC = np.loadtxt(FC_path, delimiter=",")

        # Corr_ts
        timeseries = f"{subj_dir_id}/results/corrlabel_ts.txt"
        timeseries = np.loadtxt(timeseries)
        timeseries = timeseries.T
    except OSError:
        # això es suficient per fer les que estan completes?
        print(f"{subID}_{type_dir} not found!")
        return df_G

    # patillada pero gl
    idx_G = len(df_G) - 1

    df_G.at[idx_G, "SubjID"] = subID
    df_G.at[idx_G, "CENTER"] = type_dir

    ## GENERAL metastability
    # if its simulated, need to bandpass
    metast = kuromoto_metastability(remove_mean(timeseries, axis=1))
    df_G.at[idx_G, "Meta"] = metast

    # ONLY CORTICAL
    # need to add a flag when it is cortical and not cortical
    if cortical:
        FC_left = FC[np.ix_(np.r_[14:45], np.r_[14:45])]
        FC_right = FC[np.ix_(np.r_[45:76], np.r_[45:76])]
        FC_inter = FC[np.ix_(np.r_[14:45], np.r_[45:76])]
    else:
        FC_left = FC[np.ix_(np.r_[0:7, 14:45], np.r_[0:7, 14:45])]
        FC_right = FC[np.ix_(np.r_[7:14, 45:76], np.r_[7:14, 45:76])]
        FC_inter = FC[np.ix_(np.r_[0:7, 14:45], np.r_[7:14, 45:76])]

    FC_Corr_intra_L = np.mean(FC_left[np.triu_indices(FC_left.shape[0], 1)])
    FC_Corr_intra_R = np.mean(FC_right[np.triu_indices(FC_right.shape[0], 1)])

    # ONLY homotopic connections
    FC_Corr_inter = np.mean([FC_inter[i, i] for i in range(len(FC_inter))])

    df_G.at[idx_G, "FC_Corr_total"] = FC_Corr_inter
    df_G.at[idx_G, "FC_Corr_intra_L"] = FC_Corr_intra_L
    df_G.at[idx_G, "FC_Corr_intra_R"] = FC_Corr_intra_R

    ## Entropy
    df_G.at[idx_G, "Entropy_total"] = entropy_FC(FC)
    df_G.at[idx_G, "Entropy_L"] = entropy_FC(FC_left)
    df_G.at[idx_G, "Entropy_R"] = entropy_FC(FC_right)

    ## Integration
    df_G.at[idx_G, "Integration_total"] = integration_AUC(FC)
    df_G.at[idx_G, "Integration_L"] = integration_AUC(FC_left)
    df_G.at[idx_G, "Integration_R"] = integration_AUC(FC_right)

    # threshold and create graph
    FC = np.where(FC > th, 1, 0)
    FC_left = np.where(FC_left > th, 1, 0)
    FC_right = np.where(FC_right > th, 1, 0)

    FC_left_nx = nx.from_numpy_matrix(FC_left)
    FC_right_nx = nx.from_numpy_matrix(FC_right)
    FC_nx = nx.from_numpy_matrix(FC)

    # iterate over
    list_of_graphs = [
        FC_left_nx,
        FC_right_nx,
        FC_nx,
    ]  # , FC_left_nx, FC_right_nx, FC_inter_nx] #, Tract_ntx] #, FC_ntx]
    list_of_names = ["FC_L", "FC_R", "FC"]  # , "FC_R", "FC_inter"] #, "Tract"] #, "FC"]
    list_of_pairs = [
        permutations(FC_left_nx, 2),
        permutations(FC_right_nx, 2),
        permutations(FC_nx, 2),
    ]  # , pairs_L, pairs_R, pairs_inter]

    # iterate over the existing graphs
    for (graph, name, pairs) in zip(list_of_graphs, list_of_names, list_of_pairs):
        largest_cc = max(nx.connected_components(graph), key=len)
        G = graph.subgraph(largest_cc).copy()

        # Avg efficiency
        df_G.at[idx_G, f"{name}_avg_spl"] = nx.global_efficiency(G)

        # mean shortest path length
        df_G.at[idx_G, f"{name}_efficiency"] = nx.average_shortest_path_length(G)

    # return (df_G, df_nodes)
    return df_G


@click.command(
    help="Run over the existing subjects, load the networks and extract their values."
)
@click.option(
    "--total_csv",
    required=True,
    type=click.STRING,
    help="csv with the base information for every subject",
)
@click.option(
    "--pip_csv",
    required=True,
    type=click.STRING,
    help="csv with the current pipeline information for every subject",
)
@click.option(
    "--out_csv_prefix",
    required=True,
    type=click.STRING,
    help="Output csv prefix. Will output various csv files",
)
@click.option("--njobs", required=True, type=click.STRING, help="number of jobs")
@click.option("--th", required=True, type=click.FLOAT, help="threshold to binarize")
@click.option("--cortical", is_flag=True, help="use only cortical values.")
@click.argument("subj_dir")
def compute_FC_values(
    subj_dir, total_csv, pip_csv, out_csv_prefix, njobs, th, cortical
):
    """
    Compute FC values
    """

    # iterate over the subjects
    df_total = pd.read_csv(total_csv)
    df_pipeline = pd.read_csv(pip_csv)

    # will save everything in dictionaries to save it later to df. Columns are labels
    df_G = pd.DataFrame()
    # df_nodes = pd.DataFrame()

    njobs = int(njobs)

    # load the study

    # at least dt status, so that we have processed lesions volumes
    # HACK: SELECT ONLY 5 SUBJECTS, TO TEST
    results = Parallel(n_jobs=njobs, backend="threading")(
        delayed(par_extract_values)(row, subj_dir, th, cortical)
        for row in df_total.itertuples()
        if df_pipeline[
            (df_pipeline.id == row.SubjID) & (df_pipeline.CENTER == row.CENTER)
        ]["agg_SC"].bool()
        & df_pipeline[
            (df_pipeline.id == row.SubjID) & (df_pipeline.CENTER == row.CENTER)
        ]["fMRI"].bool()
    )

    list_of_G = [G for G in results]
    # list_of_nodes = [nodes for (_, nodes) in results]

    df_G = pd.concat(list_of_G)
    # df_nodes = pd.concat(list_of_nodes)

    # save to csv
    if cortical:
        df_G.to_csv(f"{out_csv_prefix}_G_FC_cort.csv")
    else:
        df_G.to_csv(f"{out_csv_prefix}_G_FC.csv")
    # df_nodes.to_csv(f'{out_csv_prefix}_nodes_SC.csv')


if __name__ == "__main__":
    # those parameters have to be entered from outside
    compute_FC_values()
