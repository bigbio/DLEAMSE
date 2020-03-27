# DLEAMSE
A Deep LEArning-based Mass Spectra Embedder for spectral similarity scoring. 
  
DLEAMSE (based on Siamese Network) is trained and tested with a larger dataset from PRIDE Cluster. The repository stores the encoder and embedder scripts of DLEAMSE to encode and embed spectra.

# Training data set

A larger spectral set from PRIDE Cluster is used to construct the training and test data, which use high confidence spectra retrieved from high consistency clusters. We chose PRIDE Cluster data to train and test DLEAMSE, for two reasons: 1. The spectra in high consistency clusters are high confidence spectra. 2. The spectral set from PRIDE Cluster covers more species and instrument types. Two filters were used for retrieving high confidence spectra. The first filter controls the quality of collected clusters. We customized clustering-file-converter (https://github.com/spectra-cluster/clustering-file-converter) to retain the high-quality spectral clusters (cluster size >= 30, cluster ratio >= 0.8, and the total ions current (TIC) >= 0.2). The second filter eliminates duplicate clusters assigned with same peptide sequence, only one in the dupli-cates has been chosen, to ensure that the retained clusters are from different peptides. Then 113,362 clusters have been retrained from PRIDE Cluster release 201504. The needed spectra in clusters are acquired from the PRIDE Archive.

# Model and Training

In DLEAMSE, Siamese network (Figure 1a) trains two same embedding models (Figure 1c) with shared weights, and spectra are encoded by the same encoder (Figure 1b) before the embedding. Based on the Euclidean distance between the pair of embedded spectra, the weights of embedding model is learned by contrastive loss function adapted from Hadsell et. al. that penalizes far-apart same-label spectra (label=1) and nearby different-label spectra (label=0). Back propagation from the loss function is used to update the weights in the network. The net-work is trained by stochastic gradient descent with the Adam update rule with a learning rate of 0.005. The codes are implemented in Python3 with the PyTorch framework.
![model](https://github.com/bigbio/DLEAMSE/blob/master/src/DLEAMSE/dleamse_modle_references/model.png)

# Testing
![loss and test](https://github.com/bigbio/DLEAMSE/blob/master/src/DLEAMSE/dleamse_modle_references/loss_and_test.png)

# Requirements

- Python3.7 (or Anaconda3)
- torch==1.0.0 (python -m pip install torch===1.0.0 torchvision===0.2.1 -f https://download.pytorch.org/whl/torch_stable.html)
- pyteomics>=3.5.1
- numpy>=1.13.3
- numba>=0.45.0
- faiss-gpu==1.5.3 (if you want to use faiss index making and searching function)    
- more_itertools==7.1.0

# Installation

DLEAMSE’s encoder and embedder have been packaged and uploaded to pypi library, the package’s name is [dleamse](https://pypi.org/project/dleamse/).

`python -m pip install dleamse`

# Usage

The model file of DLEAMSE: [080802_20_1000_NM500R_model.pkl](https://github.com/bigbio/DLEAMSE/tree/master/src/DLEAMSE/siamese_modle_reference)
The 500 reference spectra used in our project: [500_rfs_spectra.mgf](https://github.com/bigbio/DLEAMSE/tree/master/src/DLEAMSE/siamese_modle_reference)

## Encode and Embed spectra, then write faiss index

```python
# -*- coding:utf8 -*-
from dleamse.dleamse_encode_and_embed import encode_and_embed_spectra
from dleamse.dleamse_encode_and_embed import SiameseNetwork2
from dleamse.dleamse_faiss_index_writer import FaissWriteIndex

if __name__ == '__main__':
    # encode and embedded spectra
     model = "./dleamse_model_references/080802_20_1000_NM500R_model.pkl"
    prj = "test"
    input_file = "PXD003552_61576_ArchiveSpectrum.json"
    reference_spectra = "./dleamse_model_references/0722_500_rf_spectra.mgf"
    output_embedd_file = "PXD003552_61576_ArchiveSpectrum_embedded.npy"

    embedded_spectra_data = encode_and_embed_spectra(model, prj, input_file, reference_spectra, output_embedded_file)

    # faiss index writer
    index_ids_save_file = "index_ids_save,txt"
    index_save_file = "test_0325.index"

    index_writer = FaissWriteIndex()
    index_writer.create_index(embedded_spectra_data, index_ids_save_file, index_save_file)
```


# DLEAMSE's Scripts

## **dleamse_encoder.py**:

Encode spectra to a long vectors. Return ids_usi data and encoded_spectra data. Support the spectra file with .mgf, .mzML and .json as input. By default, two or three files would be generated from this script, the spectra encoding vectors file , spectra ids_usi file and the record file of spectra with missing charge. The default directory 500 reference spectra is in dleamse_model_references directory which is under current directory.<br>
In this example, the input spectra file is *PXD003552_61576_ArchiveSpectrum.json*, and the three generated files are: *PXD003552_61576_ArchiveSpectrum_encoded.npy*; *PXD003552_61576_ArchiveSpectrum_spectrum_ids_usi.txt*; *PXD003552_61576_ArchiveSpectrum_missing_c_record.txt* (if exist the charge missing spectra) <br>
```python
from dleamse.dleamse_encoder import encode_spectra

def test_encoder():
    #encode spectra
    prj = "test"
    input_file = "PXD003552_61576_ArchiveSpectrum.json"
    reference_spectra = "./dleamse_model_references/0722_500_rf_spectra.mgf"
    ids_usi_data, encoded_vstack_data = encode_spectra(prj, input_file, reference_spectra)
```

## **dleamse_embeder.py**:

Embed encoded_spectra data to 32D vectors. the default directory of DLEASME isin dleamse_model_references directory which is under current directory.<br>
In this example, the input encoded_spectra file is *PXD003552_61576_ArchiveSpectrum_encoded.npy, its ids_usi data is in PXD003552_61576_ArchiveSpectrum_spectrum_ids_usi.txt; and the generated  32D vectors files is : *PXD003552_61576_ArchiveSpectrum_embedded.npy <br>
```python
from dleamse.dleamse_encode_and_embed import SiameseNetwork2
from dleamse.dleamse_embeder import embed_spectra

def test_embeder():
    # embed encoded_spectra data
    model = "./dleamse_model_references/080802_20_1000_NM500R_model.pkl"
    ids_usi_data = "PXD003552_61576_ArchiveSpectrum_spectrum_ids_usi.txt"
    vstack_encoded_spectra = "PXD003552_61576_ArchiveSpectrum_encoded.npy"
    output_embedd_file = "PXD003552_61576_ArchiveSpectrum_embedded.npy"

    embedded_vstack_data = embed_spectra(model, ids_usi_data, vstack_encoded_spectra, output_embedd_file)
```

## **dleamse_encode_and_embed.py**:

Encode and embed the spectra to vectors. This script support the spectra file with .mgf, .mzML and .json. By default, two or three files would be generated from this script, the spectra embedding vectors file , spectra usi file and the record file of spectra with missing charge. By default, GPU is used; the default directory of DLEASME model and 500 reference spectra file are in dleamse_model_references directory which is under current directory.<br>
In this example, the input spectra file is *PXD003552_61576_ArchiveSpectrum.json*, and the three generated files are: *PXD003552_61576_ArchiveSpectrum_embedded.npy*; *PXD003552_61576_ArchiveSpectrum_spectrum_usi.txt*; *PXD003552_61576_ArchiveSpectrum_miss_record.txt* (if exist the charge missing spectra) <br>
```python
from dleamse.dleamse_encode_and_embed import encode_and_embed_spectra
from dleamse.dleamse_encode_and_embed import SiameseNetwork2
def test_encode_and_embeder():
    # encode and embedded spectra
    model = "./dleamse_model_references/080802_20_1000_NM500R_model.pkl"
    prj = "test"
    input_file = "PXD003552_61576_ArchiveSpectrum.json"
    reference_spectra = "./dleamse_model_references/0722_500_rf_spectra.mgf"
    output_embedd_file = "PXD003552_61576_ArchiveSpectrum_embedded.npy"
    embedded_vstack_data = encode_and_embed_spectra(model, prj, input_file, reference_spectra, output_embedded_file)

```

## **dleamse_index_writer.py**:
```python
from dleamse.dleamse_faiss_index_writer import FaissWriteIndex

def test_index_write():
    # faiss index writer

    embedded_vstack_data = "PXD003552_61576_ArchiveSpectrum_embedded.npy"
    index_ids_save_file = "index_ids_save,txt"
    index_save_file = "test_0325.index"

    index_writer = FaissWriteIndex()
    index_writer.create_index(embedded_vstack_data, index_ids_save_file, index_save_file)

```

## **search_vectors_against_index.py**:
* **Range Search query 32D spectra vectors against spectra library's index file, Default threshold is 0.1.**:<br>
Range Search query 32D spectra vectors (*PXD003552_61576_ArchiveSpectrum_embedded.npy*) against spectra library's index file (*PXD003552_61576_ArchiveSpectrum.index*), and generate a result file (*test.csv*). Library index file (--index_file), USI file of spectral library(--index_usi_file),vectors file to be searched (-i, --input_embedded_spectra), and search result file (-o, --output) need to be specified.<br>
`python search_vectors_against_index.py --index_file=PXD003552_61576_ArchiveSpectrum.index --index_usi_file=PXD003552_61576_ArchiveSpectrum_spectra_usi.txt -i=PXD003552_61576_ArchiveSpectrum_embedded.npy -o=test.csv`
