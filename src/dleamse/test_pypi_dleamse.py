# -*- coding:utf8 -*-
from dleamse.dleamse_encode_and_embed import encode_and_embed_spectra
from dleamse.dleamse_encode_and_embed import SiameseNetwork2
from dleamse.dleamse_faiss_index_writer import FaissWriteIndex

import sys

if __name__ == '__main__':
    # encode and embedded spectra
    model = sys.argv[1]  # ./dleamse_model_references/080802_20_1000_NM500R_model.pkl
    prj = "test"
    input_file = sys.argv[2]
    reference_spectra = sys.argv[3]  # ./dleamse_model_references/0722_500_rf_spectra.mgf
    output_embedded_file = sys.argv[4]

    embedded_spectra_data = encode_and_embed_spectra(model, prj, input_file, reference_spectra, output_embedded_file)

    # faiss index writer
    index_writer = FaissWriteIndex()
    index_ids_save_file = "index_ids_save,txt"
    index_save_file = "test_0325.index"
    index_writer.create_index(embedded_spectra_data, index_ids_save_file, index_save_file)