# -*- coding:utf-8 -*-
"""
Search a file full of embedded spectra against a faiss index, and save the results to a file.
"""

import argparse
import os
import faiss
import sys
import numpy as np
import h5py

H5_MATRIX_NAME = 'MATRIX'

def commanline_args():
    """
    Declare all arguments, parse them, and return the args dict.
    Does no validation beyond the implicit validation done by argparse.
    return: a dict mapping arg names to values
    """

    # declare args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('indexfile', type=argparse.FileType('r'),
                        help='input index file')
    parser.add_argument('embedded', type=argparse.FileType('r'), nargs='+',
                        help='input embedded spectra file(s)')
    parser.add_argument('--k', type=int, help='k for kNN', default=5)
    parser.add_argument('--out', type=argparse.FileType('w'), required=True,
                        help='output file (should have extension .h5)')
    return parser.parse_args()

def read_faiss_index(index_filepath):
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

def load_embedded_spectra_vector(filepath: str) -> np.array:
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
            raise ValueError("read_array_file: Unknown extension {} for file {}".format(extension_lower, filepath))
    else:
        raise Exception('File "{}" does not exists'.format(filepath))

def search_index(index, embedded, k):
    """
    Simple search. Making this a method so I always remember to square root the results
    :param index:
    :param embedded:
    :param k:
    :return:
    """
    D, I = index.search(embedded, k)
    # search() returns squared L2 norm, so square root the results
    D = D**0.5
    return D, I

def write_search_results(D, I, outpath):
    with h5py.File(outpath, 'w') as h5f:
        h5f.create_dataset('spectrum_ids', data=np.array(range(D.shape[0])), chunks=True)
        h5f.create_dataset('D', data=D, chunks=True)
        h5f.create_dataset('I', data=I, chunks=True)

def executeSearch():

    args = commanline_args()
    print("loading index file...")
    index = read_faiss_index(args.indexfile.name)

    print("loading embedded spectra vector...")
    embedded_arrays = []
    run_spectra = load_embedded_spectra_vector(args.embedded.name)
    embedded_arrays.append(run_spectra)
    embedded_spectra = np.vstack(embedded_arrays)
    print("  Read a total of {} spectra".format(embedded_spectra.shape[0]))
    D, I = search_index(index, embedded_spectra, args.k)
    print("Writing results to {}...".format(args.out.name))
    write_search_results(D, I, args.out.name)
    print("Wrote output file.")
    args.out.close()

if __name__ == "__main__":
    executeSearch()
