# GraphNetworkMarkers
Scripts to extract and analyze meaningful values of brain SC and FC networks. Scripts used for the paper: "Conservation of hemispheric brain connectivity in patients with Multiple Sclerosis""

## Description of files

* extract_SC_values.py: python script that iterates over the existing and done subjects, load the SC matrices, and extract a .csv with the values:
* extract FC values.py: python script that iterats over existing and done subjects, load the FC matrices, and extract a csv with the values:
* compute_delays.py: compute matrices of tract lengths divided by computed cs.
* corpus_callosum.py: create a matrix with values of the corpus callosum of each subject.
* long_short_corr.py: calculate correlation of short/long pathways in the SC and FC

Folder PAPER_FIGURES_TABLES:
* contains the jupyter notebooks with the code to generate the figures and tables of the paper.

Folder jupyter_scripts:
* contains the jupyter notebooks with various experiments.

## References
Martí-Juan, G., Sastre-Garriga, J., Vidal-Jordana, A., et al. (2024). **Conservation of structural brain connectivity in
people with Multiple Sclerosis**. Network Neuroscience. DOI: [10.1162/netn_a_00404](https://doi.org/10.1162/netn_a_00404)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
The code is provided as is, and the authors do not guarantee its correctness. Data availability is restricted to data agreements between Vall d’Hebron Research Institute and each participating center. MRI and the corresponding processed data are available upon request and data transfer approval with the corresponding center.

