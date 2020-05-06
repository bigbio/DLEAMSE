# -*- coding:utf-8 -*-
"""
Search a file full of embedded spectra against a faiss index, and save the results to a file.
"""

import argparse
import ast
import json
import os
import sys

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

  def range_search(self, index_path, index_ids_usi_file, embedded, threshold, outpath="faiss_range_search_result.csv"):
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

    index_ids_usi_df = pd.read_csv(index_ids_usi_file, index_col=None)

    for i in range(embedded.shape[0]):
      query_id.append(i)
      limit_num.append(limit[i + 1] - limit[i])
      tmp_I = I[limit[i]:limit[i + 1]]
      tmp_D = D[limit[i]:limit[i + 1]]
      tmp_result = {}
      for j in range(limit[i + 1] - limit[i]):
        key = index_ids_usi_df.loc[index_ids_usi_df["ids"] == tmp_I[j]]["usi"].values[0]
        tmp_result[key] = tmp_D[j]
      result_list.append(tmp_result)

    result_df = pd.DataFrame({"query_id": query_id, "limit_num": limit_num, "result": result_list},
                             columns=["query_id", "limit_num", "result"])
    result_df.to_csv(outpath, index=False)

  def upper_range_search(self, index_path, index_ids_usi_file, embedded, lower_t, threshold, outpath="faiss_range_search_result.csv"):
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
    lower_result = index.range_search(embedded, lower_t)
    limit, D, I = result[0], result[1], result[2]
    lower_limit, lower_D, lower_I = lower_result[0], lower_result[1], lower_result[2]

    index_ids_usi_df = pd.read_csv(index_ids_usi_file, index_col=None)
    f = open(outpath, "w")

    result_list = []
    for i in range(embedded.shape[0]):

      tmp_result_dict = {}
      tmp_result_dict["query_index"] = i
      query_id.append(i)
      # limit_num.append(limit[i + 1] - limit[i])
      upper_limit_num = limit[i + 1] - limit[i]
      lower_limit_num = lower_limit[i + 1] - lower_limit[i]
      if upper_limit_num == lower_limit_num:
        tmp_I = I[limit[i]:limit[i + 1]]
        tmp_D = D[limit[i]:limit[i + 1]]
      else:
        tmp_I, tmp_D = [], []
        test_tmp_I = I[limit[i]:limit[i + 1]]
        test_tmp_D =  D[limit[i]:limit[i + 1]]
        test_l_I = lower_I[lower_limit[i]:lower_limit[i + 1]]
        test_l_D = lower_D[lower_limit[i]:lower_limit[i + 1]]
        for k in range(upper_limit_num):
          if test_tmp_I[k] in test_l_I:
            continue
          else:
            tmp_I.append(test_tmp_I[k])
            tmp_D.append(test_tmp_D[k])

      tmp_result_list = []
      for j in range(upper_limit_num - lower_limit_num):
        tmp_result = {}
        key = index_ids_usi_df.loc[index_ids_usi_df["ids"] == tmp_I[j]]["usi"].values[0]
        tmp_result["usi"] = str(key)
        tmp_result["similarity_score"] = str(tmp_D[j])
        tmp_result_list.append(tmp_result)

      tmp_result_dict["result"] = tmp_result_list

      result_list.append(tmp_result_dict)

    json_data = json.dumps(result_list)
    # print(json_data)
    f.write(json_data)
    f.close()

      # jsObject = json.dumps(result_list, cls=NumpyEncoder)
      # f = open(outpath, "w")
      # f.write(jsObject)

    # result_df = pd.DataFrame({"query_id": query_id, "limit_num": limit_num, "result": result_list},
    #                          columns=["query_id", "limit_num", "result"])
    # result_df.to_csv(outpath, index=False)

  def new_range_search(self, index_path, index_ids_usi_file, embedded, threshold, outpath="faiss_range_search_result.csv"):
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

    print("The number of query embedded spectra : {}".format(embedded.shape[0]))
    result = index.range_search(embedded, dist)

    limit, D, I = result[0], result[1], result[2]

    index_ids_usi_df = pd.read_csv(index_ids_usi_file, index_col=None)
    f = open(outpath, "w")

    result_list = []
    for i in range(embedded.shape[0]):

      tmp_result_dict = {}
      tmp_result_dict["query_index"] = i
      tmp_I = I[limit[i]:limit[i + 1]]
      tmp_D = D[limit[i]:limit[i + 1]]

      tmp_result_list = []
      for j in range(limit[i + 1] - limit[i]):
        tmp_result = {}
        key = index_ids_usi_df.loc[index_ids_usi_df["ids"] == tmp_I[j]]["usi"].values[0]
        tmp_result["usi"] = str(key)
        tmp_result["similarity_score"] = str(tmp_D[j])
        tmp_result_list.append(tmp_result)

      tmp_result_dict["result"] = tmp_result_list

      result_list.append(tmp_result_dict)

    json_data = json.dumps(result_list)
    f.write(json_data)
    f.close()

  def execute_range_search(self, index_file, index_ids_usi_file, embedded_spectra_file, lower_threshold, upper_threshold, output_file):

    print("loading embedded spectra vector...")
    # embedded_arrays = []
    run_spectra = self.load_embedded_spectra_vector(embedded_spectra_file)
    # embedded_arrays.append(run_spectra)
    # embedded_spectra = np.vstack(embedded_arrays)
    print("  Read a total of {} spectra".format(run_spectra.shape[0]))
    if lower_threshold == 0 or lower_threshold == upper_threshold:
      print("Runing range search ...")
      self.new_range_search(index_file, index_ids_usi_file, run_spectra.astype('float32'), threshold, output_file)
      print("Wrote results to {}...".format(output_file))
    elif 0 < lower_threshold < upper_threshold:
      print("Runing upper range search ...")
      self.upper_range_search(index_file, index_ids_usi_file, run_spectra.astype('float32'), lower_threshold, threshold, output_file)
      print("Wrote results to {}...".format(output_file))
    else:
      print("Wrong lower_threshold value, please enter a correct threshold value.")


if __name__ == "__main__":
  index_file = sys.argv[1]
  index_ids_usi_file = sys.argv[2]
  embedded_spectra = sys.argv[3]
  threshold = 0.07
  output = sys.argv[4]

  index_searcher = FaissIndexSearch()
  index_searcher.execute_range_search(index_file, index_ids_usi_file, embedded_spectra, threshold, output)

  # data = [{'a':1, 'b':2, 'c':[{'a1':1, 'b1':2, 'c1':3}, {'a2':1, 'b2':2, 'c2':3}]}, {'a1':1, 'b1':2, 'c1':3}]
  # json_data = json.dumps(data)
  # print(json_data)
