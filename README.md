# DLEAMSE
A Deep LEArning-based Mass Spectra Embedder for spectral similarity scoring. 
  
DLEAMSE (based on Siamese Network) is trained and tested with a larger dataset from PRIDE Cluster. The repository stores the encoder and embedder scripts of DLEAMSE to encode and embed spectra.

# Training data set
A larger spectral set from PRIDE Cluster is used to construct the training and test data, which use high confidence spectra retrieved from high consistency clusters. We chose PRIDE Cluster data to train and test DLEAMSE, for two reasons: 1. The spectra in high consistency clusters are high confidence spectra. 2. The spectral set from PRIDE Cluster covers more species and instrument types. Two filters were used for retrieving high confidence spectra. The first filter controls the quality of collected clusters. We customized clustering-file-converter (https://github.com/spectra-cluster/clustering-file-converter) to retain the high-quality spectral clusters (cluster size >= 30, cluster ratio >= 0.8, and the total ions current (TIC) >= 0.2). The second filter eliminates duplicate clusters assigned with same peptide sequence, only one in the dupli-cates has been chosen, to ensure that the retained clusters are from different peptides. Then 113,362 clusters have been retrained from PRIDE Cluster release 201504. The needed spectra in clusters are acquired from the PRIDE Archive.

# Model and Training
In DLEAMSE, Siamese network (Figure 1a) trains two same embedding models (Figure 1c) with shared weights, and spectra are encoded by the same encoder (Figure 1b) before the embedding. Based on the Euclidean distance between the pair of embedded spectra, the weights of embedding model is learned by contrastive loss function adapted from Hadsell et. al. that penalizes far-apart same-label spectra (label=1) and nearby different-label spectra (label=0). Back propagation from the loss function is used to update the weights in the network. The net-work is trained by stochastic gradient descent with the Adam update rule with a learning rate of 0.005. The codes are implemented in Python3 with the PyTorch framework.
![model](https://github.com/qinchunyuan/DLEAMSE/blob/master/src/DLEAMSE/dleamse_modle_references/model.png)

# Testing
![loss and test](https://github.com/qinchunyuan/DLEAMSE/blob/master/src/DLEAMSE/dleamse_modle_references/loss_and_test.jpg)

# Requirements

* Python3 (or Anaconda3)

* torch-1.0.0 (cpu or gpu version)

* pyteomics>=3.5.1

* numpy>=1.13.3

* numba>=0.45.0

# Installation
DLEAMSE’s encoder and embedder have been packaged and uploaded to pypi library, the package’s name is [dleamse](https://pypi.org/project/dleamse/).

`pip3 install dleamse`

# Usage
The model file of DLEAMSE: [080802_20_1000_NM500R_model.pkl](https://github.com/bigbio/DLEAMSE/tree/master/src/DLEAMSE/siamese_modle_reference)
The 500 reference spectra used in our project: [500_rfs_spectra.mgf](https://github.com/bigbio/DLEAMSE/tree/master/src/DLEAMSE/siamese_modle_reference)

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

# Command Line Scripts

* **encode_and_embed.py**: The commandline script for encoding and embedding spectra, usi is added for each output 32d vector.
** &nbsp;&nbsp;a)python encode_and_embed.py -i=PXD003552_61576_ArchiveSpectrum.json
*** &nbsp;&nbsp;&nbsp;&nbsp; i.	Encode and embed spectra to 32D vectors. Encode and embed the spectra (PXD003552_61576_ArchiveSpectrum.json) to vectors.
**** &nbsp;&nbsp;&nbsp;&nbsp; ii.By default, the generated spectra embedding vectors file (PXD003552_61576_ArchiveSpectrum.npy), the file of the spectra USI data (PXD003552_61576_ArchiveSpectrum_spectrum_usi.txt) and record file of the spectra with missing charge (PXD003552_61576_ArchiveSpectrum_miss_record. txt) are stored in current directory.
** &nbsp;&nbsp;b) python encode_and_embed.py -i=PXD003552_61576_ArchiveSpectrum.json --make_faiss_index=True.
*** &nbsp;&nbsp;&nbsp;&nbsp; i.	Make index for spectral library. Eencode and embed spectra to 32D vectors, then make the faiss index file for these vectors: if you want to build the index after the encoding embedding, use the setting --make_faiss_index=True.
*** &nbsp;&nbsp;&nbsp;&nbsp; ii.The same as the explanation in 1.a)ii; in addition, the default value of the --make_faiss_index parameter is False. If you want to use the index building function, you need to set it to True.
* **search_vectors_against_index.py**<
** &nbsp;&nbsp;a)python search_vectors_against_index.py --index_file=PXD003552_61576_ArchiveSpectrum.index -i=PXD003552_61576_ArchiveSpectrum_embedded.npy -o=./test.h5
*** &nbsp;&nbsp;&nbsp;&nbsp; i.	Search query 32D spectra vectors (PXD003552_61576_ArchiveSpectrum_embedded.npy) against spectra library's index file (PXD003552_61576_ArchiveSpectrum.index), and generate a result file (test.h5).
*** &nbsp;&nbsp;&nbsp;&nbsp; ii.KNN algorithm is used by default, k = 5; library index file (--index file), vectors file to be searched (-i, --input_embedded_spectra), and search result file (-o, --output) need to be specified.