# -*- coding:utf-8 -*-
"""
Search a file full of embedded spectra against a faiss index, and save the results to a file.
"""

import argparse
import ast
import os
import faiss
import numpy as np
import h5py
import pandas as pd

H5_MATRIX_NAME = 'MATRIX'
DEFAULT_IVF_NLIST = 100


class FaissIndexSearch:

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
        :param filepath: file path to write the file
        :return:
        """

    if os.path.exists(filepath):
      extension_lower = filepath[filepath.rfind("."):].lower()
      if extension_lower == ".txt":
        ids_embedded_spectra = pd.read_csv(filepath, sep="\t", index_col=None)
        # ids_data = ids_embedded_spectra["ids"].values
        spectra_vectors = ids_embedded_spectra["embedded_spectra"].values
        tmp_spectra_vectors = None
        if type(spectra_vectors[0]) == type("test"):
          tmp_data = []
          for vec in spectra_vectors:
            tmp_data.append(ast.literal_eval(vec))
          tmp_spectra_vectors = np.vstack(tmp_data)
          tmp_data.clear()
        vectors = tmp_spectra_vectors
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

  def write_knn_search_results(self, d, i, outpath):
    with h5py.File(outpath, 'w') as h5f:
      h5f.create_dataset('spectrum_ids', data=np.array(range(d.shape[0])), chunks=True)
      h5f.create_dataset('D', data=d, chunks=True)
      h5f.create_dataset('I', data=i, chunks=True)

  def execute_knn_search(self, index_file, embedded_spectra_file, k, output_file):

    index = self.read_faiss_index(index_file)

    print("loading embedded spectra vector...")
    embedded_arrays = []
    run_spectra = self.load_embedded_spectra_vector(embedded_spectra_file)
    embedded_arrays.append(run_spectra)
    embedded_spectra = np.vstack(embedded_arrays)
    print("  Read a total of {} spectra".format(embedded_spectra.shape[0]))
    D, I = self.knn_search(index, embedded_spectra.astype('float32'), k)
    self.write_knn_search_results(D, I, output_file)
    print("Wrote results to {}...".format(output_file))

  def range_search(self, index_path, embedded, threshold, outpath="faiss_range_search_result.csv"):
    """
        Range Search can only use in CPU
        :param outpath:
        :param threshold: similarity thershold
        :param index_path: index path
        :param embedded: embeded file
        :return:
        """
    print("loading index file...")
    index = faiss.read_index(index_path)  # cpu
    dist = threshold  # Threshold
    query_id, limit_num, result_list = [], [], []
    print("The number of query embedded spectra : {}".format(embedded.shape[0]))
    result = index.range_search(embedded, dist)
    limit, D, I = result[0], result[1], result[2]

    for i in range(embedded.shape[0]):
      query_id.append(i)
      limit_num.append(limit[i + 1] - limit[i])
      tmp_I = I[limit[i]:limit[i + 1]]
      tmp_D = D[limit[i]:limit[i + 1]]
      tmp_result = {}
      for i in range(limit[i + 1] - limit[i]):
        tmp_result[tmp_I[i]] = tmp_D[i]
      result_list.append(tmp_result)

    result_df = pd.DataFrame({"query_id": query_id, "limit_num": limit_num, "result": result_list},
                             columns=["query_id", "limit_num", "result"])
    result_df.to_csv(outpath, index=False)

  def execute_range_search(self, index_file, embedded_spectra_file, threshold, output_file):

    print("loading embedded spectra vector...")
    # embedded_arrays = []
    run_spectra = self.load_embedded_spectra_vector(embedded_spectra_file)
    # embedded_arrays.append(run_spectra)
    # embedded_spectra = np.vstack(embedded_arrays)
    print("  Read a total of {} spectra".format(run_spectra.shape[0]))

    self.range_search(index_file, run_spectra.astype('float32'), threshold, output_file)
    print("Wrote results to {}...".format(output_file))
