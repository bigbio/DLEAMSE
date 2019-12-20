# DLEAMSE
A Deep LEArning-based Mass Spectra Embedder for spectral similarity scoring. 
  
DLEAMSE (based on Siamese Network) is trained and tested with a larger dataset from PRIDE Cluster. 

# Requirements

* Python3 (or Anaconda3)

* torch-1.0.0 (cpu or gpu version)

* pyteomics>=3.5.1

* numpy>=1.13.3

* numba>=0.45.0

# Installation

`pip3 install dleamse`

# Usage

## 1. Encode spectra

```python
from dleamse.dleamse_encoder import encode_spectra
if __name__ == "__main__":
	encoded_spectra_data = encode_spectra("input.mgf", "500rf_spectra.mgf", "cmiss_record.txt","./encodes_result.txt")
```

## 2. Embed spectra from encoded_spectra file

```python
from dleamse.dleamse_embeder import embed_spectra
from dleamse.dleamse_embeder import SiameseNetwork2

if __name__ == "__main__":
	model = "model_file.pkl"
	embedded_spectra_data = embed_spectra(model, encoded_spectra_data,"embedded_result.csv", use_gpu=False)
```

# Scripts
  1. **useFADLEAMSE.py**: encode and embed spectra, take one file (.mgf) as input and output a .csv file which contains 32d vectors.
  2. **ndp_usetime.py**: calculate computing time of normalized dot product (square-root tansformed, intensity normalization, top 100 peaks), and use @njit accelaration.
  3. **dleamse_usetime_cpu.py**: calculate conputing time of dleamse based similarity scoring with CPU, use @njit accelaration.
  4. **dleamse_usetime_gpu.py**: calculate computing time of dleamse based similarity socring with GPU, use @njit accelaration.

# Example
 1. python useFASLEAMSE.py ../siamese_modle_reference/080802_20_1000_NM500R_model.pkl --input ./data/130402_08.mgf --output ./data/test.csv





