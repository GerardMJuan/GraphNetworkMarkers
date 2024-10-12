"""
Script to extract values of the corpus callosumfor each subject

Create new dataframe with the values, and ICV.

Probably there is a tool to extract them (fs)
"""

import numpy as np
import os
import click
import subprocess
import numpy as np
import datetime
import pandas as pd

@click.command(help="Run over the existing subjects, load the networks and extract their values.")
@click.option("--total_csv", required=True, type=click.STRING, help="csv with the base information for every subject")
@click.option("--pip_csv", required=True, type=click.STRING, help="csv with the current pipeline information for every subject")
@click.option("--out_csv", required=True, type=click.STRING, help="Output csv.")
@click.argument("subj_dir")
def compute_corpus_callosum(subj_dir, total_csv, pip_csv, out_csv):
    """
    Compute corpus callosum values

    use asegstats2table for each subject to extract the txt values, load them into a dataframe
    get the corpus callosum and save those values
    """

    # iterate over the subjects
    df_total = pd.read_csv(total_csv)
    df_pipeline = pd.read_csv(pip_csv)

    list_of_df = []

    # at least dt status, so that we have processed lesions volumes
    for row in df_total.itertuples():
        type_dir = row.CENTER
        subID = row.SubjID
        status = df_pipeline[(df_pipeline.id==subID) & (df_pipeline.CENTER==type_dir)]["fastsurfer"].bool()
        if status:

            subj_dir_id = f'{subj_dir}/{type_dir}_Post/{subID}'
            print(subj_dir_id)
            ### MIDSAGITTAL PLANE
            # REGISTER TO COMMON MNI305
            # os.system(f"mri_vol2vol --mov {subj_dir_id}/recon_all/mri/aseg.mgz --targ $FREESURFER_HOME/average/mni305.cor.mgz  --xfm {subj_dir_id}/recon_all/mri/transforms/talairach.xfm --o {subj_dir_id}/recon_all/mri/aseg_mni305.mgz --interp nearest")

            # convert to nii.gz to work with 
            # os.system(f"mri_convert {subj_dir_id}/recon_all/mri/aseg_mni305.mgz {subj_dir_id}/recon_all/mri/aseg_mni305.nii.gz")

            # cut the midsagital slice (we assume that the previous registration have put the scan in a correct space to make this vertical cut)
            # os.system(f"fslmaths {subj_dir_id}/recon_all/mri/aseg_mni305.nii.gz -roi 127 1 0 -1 0 -1 0 -1 {subj_dir_id}/recon_all/mri/aseg_mni305_slice.nii.gz")

            # select the CC labels from FreeSurferLUT (251 to 255) and binarize
            # os.system(f"fslmaths {subj_dir_id}/recon_all/mri/aseg_mni305_slice.nii.gz -thr 251 -uthr 255 -bin {subj_dir_id}/recon_all/mri/mni305_slice_CC.nii.gz")

            # convert back to orig
            os.system(f"mri_vol2vol --mov {subj_dir_id}/recon_all/mri/orig.mgz --targ {subj_dir_id}/recon_all/mri/mni305_slice_CC.nii.gz --xfm {subj_dir_id}/recon_all/mri/transforms/talairach.xfm --inv --o {subj_dir_id}/recon_all/mri/aseg_cc_slice_orig.nii.gz --interp nearest")

            # Compute volume of mask. its the volume, but as one of the dimensions is 1mm3, we can consider it as mm2.
            result = subprocess.check_output(f"fslstats {subj_dir_id}/recon_all/mri/aseg_cc_slice_orig.nii.gz -V", shell=True, text=True)
            CC_midsaggital_area = result.split(" ")[1]
            # call asegstats2table
            # deactivate anaconda bc we need python2.7 for this.    
            #  os.system(f"python2 $FREESURFER_HOME/bin/asegstats2table -i {subj_dir_id}/recon_all/stats/aseg.stats --meas volume --tablefile {subj_dir_id}/recon_all/stats/aseg_stats.txt >/dev/null 2>&1") 

            # load the txt as dataframe
            df_data = pd.read_csv(f"{subj_dir_id}/recon_all/stats/aseg_stats.txt", sep="\t")
            df_data["SubjID"] = subID
            df_data["CENTER"] = type_dir
            df_data["CC_Sag_area"] = CC_midsaggital_area

            # select only corpus callosum columns, icv
            columns_to_include = ['SubjID', 'CENTER', 'CC_Sag_area', 'CC_Posterior', 'CC_Mid_Posterior', 'CC_Central',
                                  'CC_Mid_Anterior', 'CC_Anterior', 'BrainSegVol', 'EstimatedTotalIntraCranialVol']

            df_data = df_data[columns_to_include]

            list_of_df.append(df_data)

    df_out = pd.concat(list_of_df)
    df_out.to_csv(out_csv, index=False)

if __name__ == "__main__":
    # those parameters have to be entered from outside
    compute_corpus_callosum()
