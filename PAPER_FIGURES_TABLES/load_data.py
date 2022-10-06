"""
Small auxiliar function to load the data
so that for all the figures and tables values
it is consistent
"""

import pandas as pd
import numpy as np
from functools import reduce

# create a function to apply it to all the dataframes
def process_data(df, cs=False):
    """
    df: full merged dataframe
    
    Remove low QC, create new variables, removes LONDON2
    """
    # get only those with QC == Y
    df = df[df["QC"] == "Y"]

    # remove LONDON2 
    df = df[df["CENTER2"] != "LONDON2"]

    # get the progressives together.
    mapping_prog = {
        "HC": "HC",
        "CIS": "CIS", 
        "RRMS": "RRMS", 
        "SPMS": "PMS", 
        "PPMS": "PMS"
    }

    df['disease'] = np.where(df['GROUP']=='HC', "HC", "MS")
    df['EDSS_group'] = np.where(df['EDSS'] < 3, "EDSS<3", "EDSS>=3")
    df['SDMT_group'] = np.where(df['SDMT'] < 40, "SDMT<40", "SDMT>=40")
    df['GROUP_prog'] = df.GROUP.map(mapping_prog)

    # TODO any extra data processing should go here to not clutter i
    df["SC_spl_full"] = ( df["SC_L_avg_spl"] + df["SC_R_avg_spl"] ) / 2
    df["SC_eff_full"] = ( df["SC_L_avg_eff"] + df["SC_R_avg_eff"] ) / 2

    df["FC_spl_full"] = ( df["FC_L_avg_spl"] + df["FC_R_avg_spl"] ) / 2
    df["FC_eff_full"] = ( df["FC_L_efficiency"] + df["FC_R_efficiency"] ) / 2
    df["FC_entropy_full"] = ( df["Entropy_L"] + df["Entropy_R"] ) / 2
    df["FC_integration_full"] = ( df["Integration_L"] + df["Integration_R"] ) / 2

    df["Full_CC"] = df["CC_Posterior"] + df["CC_Mid_Posterior"] + df["CC_Central"] + df["CC_Mid_Anterior"] + df["CC_Anterior"]
    df["Comm_ratio_approx"] = ( df["CC_Posterior"] + df["CC_Mid_Posterior"] + df["CC_Central"] + df["CC_Mid_Anterior"] + df["CC_Anterior"] ) / df["EstimatedTotalIntraCranialVol"]
    df["CC_ratio_area"] = np.sqrt(df["CC_Sag_area"]) / (df["BrainSegVol"]**(1./3.))
    df["CC_Sag_area_sqrt"] = np.log10(np.sqrt(df["CC_Sag_area"]))
    df["TIV_cubicroot"] = np.log10(df["BrainSegVol"]**(1./3.))

    return df



def load_data(root='linux'):
    """
    Load the data from the path
    """
    
    ## PATHS
    if root == 'linux':
        root = '/mnt/Bessel/Gproj/Gerard_DATA/MAGNIMS_DEFINITIVE_RESULTS'
    else:
        root = 'C:/Users/gerar/Documents/MAGNIMS_DEFINITIVE_RESULTS/'

    fc_type = "conn" # either conn or dti
    conn_type = "DeltaMeta" #only uset for conn, either "Corr" or "DeltaMeta"
    cort ="_cort" # either "_cort" or ""

    # GRAPH CONN
    # could be cort or not
    csv_FC = f'{root}/graph_values/graph_CONN_G_FC{cort}.csv'
    csv_SC = f'{root}/graph_values/graph_CONN_G_SC{cort}.csv'
    csv_longshort =  f'{root}/graph_values/long_short_conn{cort}.csv'

    ## SHARED DOCS
    csv_cc = f'{root}/graph_values/cc.csv'
    extracted_values_path  = f'{root}/extracted_values.csv'
    csv_total = f'{root}/data_total.csv'

    # shared 
    df_extracted = pd.read_csv(extracted_values_path)
    df_total = pd.read_csv(csv_total)

    # Graphs
    df_cc = pd.read_csv(csv_cc)
    df_hemis = pd.read_csv(csv_SC)
    df_hemis_FC = pd.read_csv(csv_FC)
    df_longshort = pd.read_csv(csv_longshort)

    data_frames = [df_cc, df_hemis, df_hemis_FC, df_longshort]
    df_fullgraph = reduce(lambda  left,right: pd.merge(left,right,on=["SubjID", "CENTER"]), data_frames)

    # Combine dfs
    df_final = df_total.merge(df_extracted, on=["SubjID", "CENTER"])
    df_final = df_final.merge(df_fullgraph, on=["SubjID", "CENTER"])

    df_final = process_data(df_final)

    return df_final