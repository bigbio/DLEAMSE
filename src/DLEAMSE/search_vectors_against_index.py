# -*- coding:utf-8 -*-
"""
Search a file full of embedded spectra against a faiss index, and save the results to a file.
"""

import argparse
import os
import faiss
import numpy as np
import h5py
import pandas as pd

H5_MATRIX_NAME = 'MATRIX'
DEFAULT_IVF_NLIST = 100

def commanline_args():
    """
    Declare all arguments, parse them, and return the args dict.
    Does no validation beyond the implicit validation done by argparse.
    return: a dict mapping arg names to values
    """

    # declare args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index_file', type=argparse.FileType('r'), required=True, help='input index file')
    parser.add_argument('--index_usi_file', type=argparse.FileType('r'), required=True, help='input index usi file without header')
    parser.add_argument('-i', '--input_embedded_spectra', type=argparse.FileType('r'), required=True, help='input embedded spectra file(s)')
    # parser.add_argument('--k', type=int, help='k for kNN', default=5)
    parser.add_argument('-t', '--threshold', type=float, help='Threshold for similarity searching.', default=0.1)
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), required=True,  help='output file, .csv)')
    return parser.parse_args()

class Faiss_write_index():

    def __init__(self, vectors_data, output_path):
        self.tmp = None
        self.spectra_vectors = vectors_data
        self.create_index(vectors_data, output_path)

    def create_index(self, spectra_vectors, output_path):
        n_embedded_dim = spectra_vectors.shape[1]
        index = self.make_faiss_index(n_embedded_dim)
        index.add(self.spectra_vectors.astype('float32'))
        self.write_faiss_index(index, output_path)

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

class Faiss_Index_Search():
    def __init__(self):
        print("Start Faiss Index Simialrity Searching ...")

    def read_faiss_index(self, index_filepath):
        """
        Load a FAISS index. If we're on GPU, then convert it to GPU index
        :param index_filepath:
        :return:
        """
        print("read_faiss_index start.")
        index = faiss.read_index(index_filepath)
        if faiss.get_num_gpus():
            print("read_faiss_index: Converting FAISS index from CPU to GPU.")
            index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
        return index

    def load_embedded_spectra_vector(self, filepath: str) -> np.array:
        """
        load embedded vectors from input file
        :param path: the input file type is .txt or the h5 with vectors in it
        :return:
        """

        if os.path.exists(filepath):

            extension_lower = filepath[filepath.rfind("."):].lower()
            if extension_lower == '.h5':
                h5f = None
                try:
                    h5f = h5py.File(filepath, 'r')
                    result = h5f[H5_MATRIX_NAME][:]
                    return result
                except Exception as e:
                    print("Failed to read array named {} from file {}".format(H5_MATRIX_NAME, filepath))
                    raise e
                finally:
                    if h5f:
                        h5f.close()
            elif extension_lower == '.npy':
                return np.load(filepath)
            elif extension_lower == ".txt":
                vectors = np.loadtxt(filepath)
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
                return vectors
            else:
                raise ValueError(
                    "read_array_file: Unknown extension {} for file {}".format(extension_lower, filepath))
        else:
            raise Exception('File "{}" does not exists'.format(filepath))

    def knn_search(self, index, embedded, k):
        """
        Simple search. Making this a method so I always remember to square root the results
        :param index:
        :param embedded:
        :param k:
        :return:
        """
        D, I = index.search(embedded, k)
        # search() returns squared L2 norm, so square root the results
        D = D ** 0.5
        return D, I

    def range_search(self, index_path, embedded, usi_data, threshold=0.1, outpath="faiss_range_search_result.csv"):
        """
        Range Search can only use in CPU
        :param output_path:
        :param index_path:
        :param embedded:
        :param usi_data:
        :return:
        """
        print("loading index file...")
        index = faiss.read_index(index_path)  #cpu
        print("loading index usi file...")
        index_usi = pd.read_csv(usi_data, header=None).values.tolist()
        dist = threshold   #Threshold
        # dist = 0.32 ** 2   #Threshold
        query_id, limit_num, result_list = [], [], []
        print(embedded.shape[0])
        for i in range(embedded.shape[0]):
            res_index = index.range_search(embedded[[i], :], dist)  # 用第50个向量查询
            # query_id.append(index_usi[i][0])
            query_id.append(i)
            limit_num.append(res_index[0][1])
            result_dict = {}
            for j in range(len(res_index[1])):
                result_dict[index_usi[int(res_index[2][j])][0]] = res_index[1][j]
            result_list.append(result_dict)
        result_df = pd.DataFrame({"query_id": query_id, "limit_num": limit_num, "result": result_list},
                                 columns=["query_id", "limit_num", "result"])
        result_df.to_csv(outpath, index=False)

    def write_search_results(self, D, I, outpath):
        with h5py.File(outpath, 'w') as h5f:
            h5f.create_dataset('spectrum_ids', data=np.array(range(D.shape[0])), chunks=True)
            h5f.create_dataset('D', data=D, chunks=True)
            h5f.create_dataset('I', data=I, chunks=True)

    def execute_knn_search(self, args):


        index = self.read_faiss_index(args.index_file.name)

        print("loading embedded spectra vector...")
        embedded_arrays = []
        run_spectra = self.load_embedded_spectra_vector(args.input_embedded_spectra.name)
        embedded_arrays.append(run_spectra)
        embedded_spectra = np.vstack(embedded_arrays)
        print("  Read a total of {} spectra".format(embedded_spectra.shape[0]))
        D, I = self.knn_search(index, embedded_spectra.astype('float32'), args.k)
        print("Writing results to {}...".format(args.output.name))
        # self.write_search_results(D, I, args.output.name)
        print("Wrote output file.")
        args.output.close()

    def execute_range_search(self, args):

        print("loading embedded spectra vector...")
        embedded_arrays = []
        run_spectra = self.load_embedded_spectra_vector(args.input_embedded_spectra.name)
        embedded_arrays.append(run_spectra)
        embedded_spectra = np.vstack(embedded_arrays)
        print("  Read a total of {} spectra".format(embedded_spectra.shape[0]))

        self.range_search(args.index_file.name, embedded_spectra.astype('float32'), args.index_usi_file.name, args.threshold, args.output.name)
        print("Writing results to {}...".format(args.output.name))
        print("Wrote output file.")
        args.output.close()

if __name__ == "__main__":
    args = commanline_args()

    index_searcher = Faiss_Index_Search()
    index_searcher.execute_range_search(args)
