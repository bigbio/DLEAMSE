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
DLEAMSE’s encoder and embedder have been packaged and uploaded to pypi library, the package’s url is https://pypi.org/project/dleamse/.

`pip3 install dleamse`

# Usage
The model file of DLEAMSE: [080802_20_1000_NM500R_model.pkl](https://github.com/qinchunyuan/DLEAMSE/tree/master/src/DLEAMSE/siamese_modle_reference)

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
  * **use_dleamse.py**: encode and embed spectra, take one file (.mgf) as input and output a .csv file which contains 32d vectors.
  * **dleamse_encoder.py**:
  * **dleamse_embedder.py**:
  * **dleamse_usetime_gpu.py**: calculate computing time of dleamse based similarity socring with GPU, use @njit accelaration.

# Example
 ** python use_dleamse.py ../siamese_modle_reference/080802_20_1000_NM500R_model.pkl --input ./data/130402_08.mgf --output ./data/test.csv





