"""Small script than extracts, for a single subject already processed in our pipeline, several intra/inter hemispheric connectivity values.
For inter:
- 1) The CC mid saggital plane is extracted.
- 2) the full CC volume.
- 3) the commisural ratio by number of fibers.
- 4) the commisural ratio by looking directly at the tracking.

For intra:
- 1) The mean shortest path length for each hemisphere.
- 2) The efficienty for each hemisphere.
"""


import os
import sys
import numpy as np
import nibabel as nib
import argparse
import glob
import pandas as pd
import json
import subprocess
from itertools import combinations, product
import networkx as nx
from itertools import permutations
from copy import deepcopy
import mrtrix3 as mrt


def get_track_values(subject_folder):
    df = pd.DataFrame()
    print("Not implemented yet.")
    # this should use: https://mrtrix.readthedocs.io/en/latest/reference/commands/tckedit.html
    # with the -include option, with the mask of the CC generated in the function get_cc_mid_saggital_plane
    # in the same space as the tractogram.
    # Then, the number of fibers that pass through the mask can be extracted, and we return the ratio compared to
    # the total number of fibers.
    return df


def global_efficiency_weighted(G, pairs):
    """
    Compute global efficiency.

    From: https://stackoverflow.com/questions/56554132/how-can-i-calculate-global-efficiency-more-efficiently
    """
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        full_sum = [
            shortest_paths[u][v]
            for u, v in pairs
            if v in shortest_paths[u] and shortest_paths[u][v] != 0
        ]
        g_eff = np.mean(full_sum)
    else:
        g_eff = 0
    return g_eff


def global_efficiency_weighted_inverse(G, pairs):
    """
    Compute global efficiency.

    From: https://stackoverflow.com/questions/56554132/how-can-i-calculate-global-efficiency-more-efficiently
    """
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        full_sum = [
            1 / shortest_paths[u][v]
            for u, v in pairs
            if v in shortest_paths[u] and shortest_paths[u][v] != 0
        ]
        g_eff = np.mean(full_sum)
    else:
        g_eff = 0
    return g_eff


def get_sc_values(subject_folder):
    # load SC and tract length matrices
    df_G = pd.DataFrame()

    ### TODO: COMPLETE THIS
    SC_path = f"{subject_folder}/results/"
    len_path = f"{subject_folder}/results/"

    SC = np.loadtxt(open(SC_path, "rb"), skiprows=0)
    SC_len = np.loadtxt(open(len_path, "rb"), skiprows=0)

    #### NORMALIZE BY LEN
    SC = SC / SC_len
    SC = np.nan_to_num(SC, nan=0, posinf=0, neginf=0)
    df_G.at[0, f"Comm_ratio"] = np.sum(SC[np.ix_(np.r_[14:45], np.r_[45:76])]) / np.sum(
        SC
    )

    SC_left = SC[np.ix_(np.r_[14:45], np.r_[14:45])]
    SC_right = SC[np.ix_(np.r_[45:76], np.r_[45:76])]
    SC_inter = SC[np.ix_(np.r_[14:45], np.r_[45:76])]
    SC = SC[np.ix_(np.r_[14:76], np.r_[14:76])]
    SC_mask_l = SC_left != 0
    SC_mask_r = SC_right != 0
    SC_mask_i = SC_inter != 0
    SC_i = SC != 0

    SC_left[SC_mask_l] = 1 / SC_left[SC_mask_l]
    SC_right[SC_mask_r] = 1 / SC_right[SC_mask_r]
    SC_inter[SC_mask_i] = 1 / SC_inter[SC_mask_i]
    SC[SC_i] = 1 / SC[SC_i]

    SC_left_nx = nx.from_numpy_matrix(SC_left)
    SC_right_nx = nx.from_numpy_matrix(SC_right)
    SC_nx = nx.from_numpy_matrix(SC)

    # iterate over
    list_of_graphs = [
        SC_left_nx,
        SC_right_nx,
        SC_nx,
    ]  # , FC_left_nx, FC_right_nx, FC_inter_nx] #, Tract_ntx] #, FC_ntx]
    list_of_names = ["SC_L", "SC_R", "SC"]  # , "FC_R", "FC_inter"] #, "Tract"] #, "FC"]
    list_of_pairs = [
        permutations(SC_left_nx, 2),
        permutations(SC_right_nx, 2),
        permutations(SC_nx, 2),
    ]  # , pairs_L, pairs_R, pairs_inter]

    # iterate over the existing graphs, later we could add FC
    for (graph, name, pairs) in zip(list_of_graphs, list_of_names, list_of_pairs):
        df_G.at[0, f"{name}_avg_spl"] = global_efficiency_weighted(
            deepcopy(graph), deepcopy(pairs)
        )
        df_G.at[0, f"{name}_avg_eff"] = global_efficiency_weighted_inverse(
            deepcopy(graph), deepcopy(pairs)
        )

    return df_G


# first argument: subject folder
def get_cc_mid_saggital_plane(subject_folder):

    # REGISTER TO COMMON MNI305
    os.system(
        f"mri_vol2vol --mov {subject_folder}/recon_all/mri/aseg.mgz --targ $FREESURFER_HOME/average/mni305.cor.mgz  --xfm {subject_folder}/recon_all/mri/transforms/talairach.xfm --o {subject_folder}/recon_all/mri/aseg_mni305.mgz --interp nearest"
    )

    # convert to nii.gz to work with
    os.system(
        f"mri_convert {subject_folder}/recon_all/mri/aseg_mni305.mgz {subject_folder}/recon_all/mri/aseg_mni305.nii.gz"
    )

    # cut the midsagital slice (we assume that the previous registration have put the scan in a correct space to make this vertical cut)
    os.system(
        f"fslmaths {subject_folder}/recon_all/mri/aseg_mni305.nii.gz -roi 127 1 0 -1 0 -1 0 -1 {subject_folder}/recon_all/mri/aseg_mni305_slice.nii.gz"
    )

    # select the CC labels from FreeSurferLUT (251 to 255) and binarize
    os.system(
        f"fslmaths {subject_folder}/recon_all/mri/aseg_mni305_slice.nii.gz -thr 251 -uthr 255 -bin {subject_folder}/recon_all/mri/mni305_slice_CC.nii.gz"
    )

    # convert back to orig
    os.system(
        f"mri_vol2vol --mov {subject_folder}/recon_all/mri/orig.mgz --targ {subject_folder}/recon_all/mri/mni305_slice_CC.nii.gz --xfm {subject_folder}/recon_all/mri/transforms/talairach.xfm --inv --o {subject_folder}/recon_all/mri/aseg_cc_slice_orig.nii.gz --interp nearest"
    )

    # Compute volume of mask. its the volume, but as one of the dimensions is 1mm3, we can consider it as mm2.
    result = subprocess.check_output(
        f"fslstats {subject_folder}/recon_all/mri/aseg_cc_slice_orig.nii.gz -V",
        shell=True,
        text=True,
    )
    CC_midsaggital_area = result.split(" ")[1]
    # call asegstats2table
    # deactivate anaconda bc we need python2.7 for this.
    os.system(
        f"python2 $FREESURFER_HOME/bin/asegstats2table -i {subject_folder}/recon_all/stats/aseg.stats --meas volume --tablefile {subject_folder}/recon_all/stats/aseg_stats.txt >/dev/null 2>&1"
    )

    # load the txt as dataframe
    df_data = pd.read_csv(f"{subject_folder}/recon_all/stats/aseg_stats.txt", sep="\t")
    df_data["CC_Sag_area"] = CC_midsaggital_area

    # select only corpus callosum columns, icv
    columns_to_include = [
        "CC_Sag_area",
        "CC_Posterior",
        "CC_Mid_Posterior",
        "CC_Central",
        "CC_Mid_Anterior",
        "CC_Anterior",
        "BrainSegVol",
        "EstimatedTotalIntraCranialVol",
    ]

    df_data = df_data[columns_to_include]
    return df_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_folder", help="The subject folder")
    parser.add_argument("output_file", help="The output file")
    args = parser.parse_args()

    # we get the subject name
    subject_name = args.subject_folder.split("/")[-1]

    # we get the subject folder
    subject_folder = args.subject_folder

    # extract subj_name from subject_folder
    subject_name = subject_folder.split("/")[-1]

    # we get the output file
    output_file = args.output_file

    # 1: we get the CC mid saggital plane, and the CC volume
    df_ccmid = get_cc_mid_saggital_plane(subject_folder)

    # 2: extract values from SC
    df_SC = get_sc_values(subject_folder)

    # 3: extract values from tracking
    df_Tr = get_track_values(subject_folder)

    # Combine all the df into the final df
    df = pd.concat([df_SC, df_ccmid, df_Tr], axis=1)

    df["Subject"] = subject_name

    df["SC_spl_full"] = (df["SC_L_avg_spl"] + df["SC_R_avg_spl"]) / 2
    df["SC_eff_full"] = (df["SC_L_avg_eff"] + df["SC_R_avg_eff"]) / 2

    df["FC_spl_full"] = (df["FC_L_avg_spl"] + df["FC_R_avg_spl"]) / 2
    df["FC_eff_full"] = (df["FC_L_efficiency"] + df["FC_R_efficiency"]) / 2
    df["FC_entropy_full"] = (df["Entropy_L"] + df["Entropy_R"]) / 2
    df["FC_integration_full"] = (df["Integration_L"] + df["Integration_R"]) / 2

    df["Full_CC"] = (
        df["CC_Posterior"]
        + df["CC_Mid_Posterior"]
        + df["CC_Central"]
        + df["CC_Mid_Anterior"]
        + df["CC_Anterior"]
    )
    df["Comm_ratio_approx"] = (
        df["CC_Posterior"]
        + df["CC_Mid_Posterior"]
        + df["CC_Central"]
        + df["CC_Mid_Anterior"]
        + df["CC_Anterior"]
    ) / df["EstimatedTotalIntraCranialVol"]
    df["CC_ratio_area"] = np.sqrt(df["CC_Sag_area"]) / (
        df["BrainSegVol"] ** (1.0 / 3.0)
    )
    df["CC_Sag_area_sqrt"] = np.log10(np.sqrt(df["CC_Sag_area"]))
    df["TIV_cubicroot"] = np.log10(df["BrainSegVol"] ** (1.0 / 3.0))

    df.save_to_csv(output_file)
