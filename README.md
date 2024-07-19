# PSDonHSR
This file contains the instructions for using the shared Python functions to reproduce the results presented in the Rahimi-Majd et al. paper.

The Cambridge Maize 2021 (CAM Maize 2021) and Maize 2021 (CAM Maize 2022) are available in the dataset directory. 

All the functions to calculate power spectral density (PSD) statists of a given (leaf) hyperspectral reflectance (HSR) sequence or a CSV file containing multiple HSR rows are represented in the 'PSD.py' file in the PSD_src directory. The full instructions for all the functions are written inside each function. 

The functions for post-processing (correlation measurements) of CAM Maize 2021 and  CAM Maize 2022 with complete instructions are in the file 'PSD_correlations.py' in the correlation_src directory. For testing these functions, the files with the results of PSD analyses are available in the same directory for corresponding data sets. Moreover, the files of the estimated parameter of HSR data (used on the correlation functions) calculated by the Prospect D model (see https://jbferet.gitlab.io/prospect/index.html) are available for the corresponding datasets. 


The required Python libraries and the tested version of them are as follows:
numpy V1.26.4
ccipy V1.13.0
pandas V2.1.4
sklearn V1.2.2


