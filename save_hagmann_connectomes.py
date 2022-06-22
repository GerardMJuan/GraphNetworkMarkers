"""
Script that saves Hagmann connectomes from the .mat file and saves them on txt easiliy readable matrices.

Then, it runs the MSP algorithm and saves into a csv the file.

Normalization of Hagmann:
Structural connectivity between pairs of regions was measured in terms of fiber density, 
defined as the number of streamlines between the two regions, normalized by the average length of the streamlines 
and average surface area of the two regions (Hagmann et al., 2008). 
"""

import numpy as np
import os
import subprocess
import numpy as np
import pandas as pd
from scipy.io import loadmat
from joblib import Parallel, delayed
from community import best_partition, modularity
from itertools import combinations, product
import networkx as nx
from itertools import permutations
import matplotlib.pyplot as plt

# load the function to extract the MSP
from extract_SC_values import global_efficiency_weighted

# load .mat
mat_name = "Connectomes.mat"
data = loadmat(mat_name)

# create dataframe where the results will be saved
df_G = pd.DataFrame()

# per accedir a les matrius s'han de fer servir els indes [0][0][0][i][0], i anar cfanviant la i
# les matrius estan concatenades a la dim 2, dim 0 i 1 son les grandaries de la matriu
# extract all the matrices, print some of them to disk to visualize them

img_dir = "Hagman_SC_png/"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)

for i in range(data["connMatrices"][0][0][0].shape[0]):
    matrix = data["connMatrices"][0][0][0][i][0]
    print(i)
    if i == 2: break

    # iterate over every matrix
    for j in range(matrix.shape[2]):
        print(j)
        if j == 0: # if it is the first matrix, save it to disk
            plt.imshow(matrix[:, :, j])
            plt.colorbar()
            plt.savefig(f"{img_dir}SC_{i}_{j}.png")
            plt.close()
        
        SC = matrix[:, :, j]
        # create new row
        new_row = {}
        new_row["matrix_size"] = SC.shape[0]

        # separate by hemispheres
        # by looking at the image, it seems like
        # it is cut at the middle
        half_point = SC.shape[0]/2
        print(half_point) # make sure that this is an int?
        half_point = int(half_point)
        # do the same procedure as in the other papers
        SC_right = SC[np.ix_(np.r_[0:half_point], np.r_[0:half_point])]
        SC_left = SC[np.ix_(np.r_[half_point:len(SC)], np.r_[half_point:len(SC)])]
        SC_inter = SC.copy()

        # Comm ratio, calculate it
        Comm_ratio = np.sum(SC_inter[np.ix_(np.r_[0:half_point], np.r_[half_point:len(SC)])])*2 / np.sum(SC_inter)
        new_row['Comm_ratio'] = Comm_ratio

        SC_mask_l = SC_left != 0
        SC_mask_r = SC_right != 0
        SC_mask_i = SC_inter != 0

        # SC_left[SC_mask_l] = SC_left[SC_mask_l] / np.amax(SC_left)
        # SC_right[SC_mask_r] = SC_right[SC_mask_r] / np.amax(SC_right)
        # SC_inter[SC_mask_i] = SC_inter[SC_mask_i] / np.amax(SC_inter)

        SC_left[SC_mask_l] = 1 / SC_left[SC_mask_l]
        SC_right[SC_mask_r] = 1 / SC_right[SC_mask_r]
        SC_inter[SC_mask_i] = 1 / SC_inter[SC_mask_i]

        SC_left_nx = nx.from_numpy_matrix(SC_left)
        SC_right_nx = nx.from_numpy_matrix(SC_right)
        SC_inter_nx = nx.from_numpy_matrix(SC_inter)

        pairs_L = [x for x in combinations([x for x in range(len(SC_left))], 2)]
        pairs_R = [x for x in combinations([x for x in range(len(SC_left))], 2)]
        pairs_inter = [x for x in product(np.r_[0:half_point], np.r_[half_point:len(SC)])]

        new_row['L_avg_spl'] = global_efficiency_weighted(SC_left_nx, pairs_L)
        new_row['R_avg_spl'] = global_efficiency_weighted(SC_right_nx, pairs_R)
        new_row['inter_avg_spl'] = global_efficiency_weighted(SC_inter_nx, pairs_inter)

        df_G = df_G.append(new_row, ignore_index=True)

df_G.to_csv("MSP_hagmann.csv")