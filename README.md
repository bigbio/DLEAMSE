# DLEAMSE
![Python package](https://github.com/bigbio/DLEAMSE/workflows/Python%20package/badge.svg?branch=master)
![Python application](https://github.com/bigbio/DLEAMSE/workflows/Python%20application/badge.svg?branch=master)

A Deep LEArning-based Mass Spectra Embedder for spectral similarity scoring. 
  
DLEAMSE (based on Siamese Network) is trained and tested with a larger dataset from PRIDE Cluster. The repository stores the encoder and embedder scripts of DLEAMSE to encode and embed spectra.


# Training data set

A larger spectral set from PRIDE Cluster is used to construct the training and test data, which use high confidence spectra retrieved from high consistency clusters. We chose PRIDE Cluster data to train and test DLEAMSE, for two reasons: 1. The spectra in high consistency clusters are high confidence spectra. 2. The spectral set from PRIDE Cluster covers more species and instrument types. Two filters were used for retrieving high confidence spectra. The first filter controls the quality of collected clusters. We customized clustering-file-converter (https://github.com/spectra-cluster/clustering-file-converter) to retain the high-quality spectral clusters (cluster size >= 30, cluster ratio >= 0.8, and the total ions current (TIC) >= 0.2). The second filter eliminates duplicate clusters assigned with same peptide sequence, only one in the dupli-cates has been chosen, to ensure that the retained clusters are from different peptides. Then 113,362 clusters have been retrained from PRIDE Cluster release 201504. The needed spectra in clusters are acquired from the PRIDE Archive.


# Model and Training

In DLEAMSE, Siamese network (Figure 1a) trains two same embedding models (Figure 1c) with shared weights, and spectra are encoded by the same encoder (Figure 1b) before the embedding. Based on the Euclidean distance between the pair of embedded spectra, the weights of embedding model is learned by contrastive loss function adapted from Hadsell et. al. that penalizes far-apart same-label spectra (label=1) and nearby different-label spectra (label=0). Back propagation from the loss function is used to update the weights in the network. The net-work is trained by stochastic gradient descent with the Adam update rule with a learning rate of 0.005. The codes are implemented in Python3 with the PyTorch framework.


![model](https://github.com/bigbio/DLEAMSE/raw/master/dleamse/dleamse_model_references/model.png)


# Testing
![loss and test](https://github.com/bigbio/DLEAMSE/raw/master/dleamse/dleamse_model_references/loss_and_test.png)

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
## tmp_mslookup.py: the commandline script of dleamse<br>

* **Encode and Embed spectra** <br>

```python
python mslookup.py embed-ms-file -i test_cml_index/PXD003552_61576_ArchiveSpectrum.json
```

* **Create index files** <br>

```python
python mslookup.py make-index -d test_cml_index/database_ids.npy -e test_cml_index/ -o test_cml_index/test_cml_0412_01.index
```

* **Merge index files**<br>

```python
python tmp_mslookup.py merge-indexes test_cml_index/*.index test_cml_index/test_cml_merge_0412.index
```

* **Range Search**<br>

```python
python tmp_mslookup.py range-search -i test_cml_index/test_cml_0412.index -es test_cml_index/*_new_ids_embedded.txt -o test_cml_index/test_cml_rangesearch_rlt.csv
```


# DLEAMSE's Scripts

## **Encode and embed spectra : dleamse_encode_and_embed.py**:

This script support the spectra file with .mgf, .mzML and .json. By default, two or three files would be generated from this script, the spectra embedding vectors file , spectra usi file and the record file of spectra with missing charge. By default, GPU is used; the default directory of DLEASME model and 500 reference spectra file are in dleamse_model_references directory which is under current directory.<br>
In this example, the input spectra file is *PXD003552_61576_ArchiveSpectrum.json*, and the three generated files are: *PXD003552_61576_ArchiveSpectrum_embedded.txt*; *PXD003552_61576_ArchiveSpectrum_spectrum_usi.txt*; *PXD003552_61576_ArchiveSpectrum_miss_record.txt* (if exist the charge missing spectra) <br>
There are three columns of the dataframe of _embedded.txt file's data, "ids, usi, embedded_spectra"<br>

## **About index : dleamse_faiss_index_writer.py**:

* **Write multiple embedded_spectra files to an index:** <br>
Write multiple embedded_spectra files to an index file(.index) and generate a file storing index's ids (.npy).<br>
A database ids file wihch is named database_ids.npy must to be keep as the parameter. The ids of all embedded_spectra will be duplicate checked with itself and database ids. If there is a file with duplicate ids, its ids will be updated and a new embedded_spectra file (_new_ids_embedded.txt) will be generated.<br>


* **Merge multiple index files:** <br>
Add multiple new embedded_spectra files to the existing raw index file (.index), and generate a new .index file and its corresponding index's ids file.
The ids of all new embedded_spectra file and the ids of raw index file will be duplicate checked. If there is a file with duplicate ID, its ID will be updated and a new file will be generated.<br>


## **About index search : dleamse_faiss_index_search.py**:
* **Range Search query 32D spectra vectors against spectra library's index file, Default threshold is 0.1.**:<br>
Based on faiss's range search method; A result file (endwith .csv) will be generated, the result's dataframe has three coloumns, "query_id, limit_num, result"
