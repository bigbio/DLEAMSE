# -*- coding:utf-8 -*-

"""
Encode and emberder spectra.
"""

import argparse
import os

import numpy as np
import faiss

DEFAULT_IVF_NLIST = 100

def _args():
    """
    Declare all arguments, parse them, and return the args dict.
    Does no validation beyond the implicit validation done by argparse.
    return: a dict mapping arg names to values
    """

    # declare args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input', type=argparse.FileType('r'), required=True, help='input spectra vectors file')
    parser.add_argument('-o', '--output', type=str, help='output vectors file, its default path is the same as input file.', default="True")

    return parser.parse_args()

class Faiss_write_index():

    def __init__(self, vectors_data, output_path):
        self.tmp = None
        if type(vectors_data) == "np.array":
            self.spectra_vectors = vectors_data
        elif str(vectors_data).endswith(".npy"):
            self.spectra_vectors = np.load(vectors_data)

        self.output = output_path

    def create_index(self):
        n_embedded_dim = self.spectra_vectors.shape[1]
        index = self.make_faiss_index(n_embedded_dim)
        index.add(self.spectra_vectors.astype('float32'))
        self.write_faiss_index(index, self.output)
        print("Finish!")

    def make_faiss_index(self, n_dimensions, index_type='flat'):
        """
        Make a fairly general-purpose FAISS index
        :param n_dimensions:
        :param index_type: Type of index to build: flat or ivfflat. ivfflat is much faster.
        :return:
        """
        print("Making index of type {}".format(index_type))
        if faiss.get_num_gpus():
            gpu_resources = faiss.StandardGpuResources()
            if index_type == 'flat':
                config = faiss.GpuIndexFlatConfig()
                index = faiss.GpuIndexFlatL2(gpu_resources, n_dimensions, config)
            elif index_type == 'ivfflat':
                config = faiss.GpuIndexIVFFlatConfig()
                index = faiss.GpuIndexIVFFlat(gpu_resources, n_dimensions, DEFAULT_IVF_NLIST, faiss.METRIC_L2, config)
            else:
                raise ValueError("Unknown index_type %s" % index_type)
        else:
            print("Using CPU.")
            if index_type == 'flat':
                index = faiss.IndexFlatL2(n_dimensions)
            elif index_type == 'ivfflat':
                quantizer = faiss.IndexFlatL2(n_dimensions)
                index = faiss.IndexIVFFlat(quantizer, n_dimensions, DEFAULT_IVF_NLIST, faiss.METRIC_L2)
            else:
                raise ValueError("Unknown index_type %s" % index_type)
        return index

    def write_faiss_index(self, index, out_filepath):
        """
        Save a FAISS index. If we're on GPU, have to convert to CPU index first
        :param index:
        :return:
        """
        if faiss.get_num_gpus():
            print("Converting index from GPU to CPU...")
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, out_filepath)
        print("Wrote FAISS index to {}".format(out_filepath))

if __name__ == '__main__':
    # python encode_and_embed.py --input="./dleamse_model_references/CHPP_LM3_RP10_1.mzML" --output="0211_test_output.txt"

    args = _args()
    input_file = args.input.name
    output = args.output
    output_file = None
    if output:
        dirname, filename = os.path.split(os.path.abspath(input_file))
        output_file = dirname + "/" + filename.strip(".npy") + ".index"

    index_maker = Faiss_write_index(input_file, output_file)
    index_maker.create_index()

