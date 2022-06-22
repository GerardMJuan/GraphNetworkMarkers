# GraphNetworkMarkers
Scripts to extract meaningful values of brain SC and FC networks.

## Description of files

* extract_SC_values.py: python script that iterates over the existing and done subjects, load the SC matrices, and extract a .csv with the values:
  * Weighted clustering coefficient
  * Average connectivity (TODO: thresholding?)
  * Global efficiency
  * Shortest path length average
  * Centrality (do we need thresholding?)
  * Eigenvector centrality
  * Modularity
  * Small world (omega)

* extract FC values.py: python script that iterats over existing and done subjects, load the FC matrices, and extract a csv with the values:

* compute_delays.py: compute matrices of tract lengths divided by computed cs.
* generate_smaller_mat.py: generates smaller matrices: separated by hemispheres, and by lobes (not implemented yet)
* corpus_callosum.py: create a matrix with values of the corpus callosum of each subject.
* 

## Credits
Me
And the creators of the codes and packages used here: