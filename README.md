# DLEAMSE
A Deep LEArning-based Mass Spectra Embedder for spectral similarity scoring. 
  
DLEAMSE (based on Siamese Network) is trained and tested with a larger dataset from PRIDE Cluster. 

# Requirements
Python3 (Anaconda3)    
torch-1.0.0 (gpu or cpu version)    
pyteomics-3.5.1    
numpy-1.13.3    
numba-0.45.0

# Scripts
  1. **useFADLEAMSE.py**: encode and embed spectra, take one file (.mgf) as input and output a .csv file which contains 32d vectors.
  2. **ndp_usetime.py**: calculate computing time of normalized dot product (square-root tansformed, intensity normalization, top 100 peaks), and use @njit accelaration.
  3. **dleamse_usetime_cpu**: calculate conputing time of dleamse based similarity scoring with cpu, use @njit accelaration.
  4. **dleamse_usetime_gpu**: calculate computing time of dleamse based similarity socring with GPU, use @njit accelaration.
  
# Example
 1. python useFASLEAMSE.py ../siamese_modle_reference/080802_20_1000_NM500R_model.pkl --input ./data/130402_08.mgf --output ./data/test.csv


