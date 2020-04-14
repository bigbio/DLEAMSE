# -*- coding:utf-8 -*-
import os
import sys
import unittest

import numpy as np
import pandas as pd
import faiss
import ast
import click

DEFAULT_IVF_NLIST = 100


class FaissWriteIndex:

  def __init__(self):
    self.tmp = None
    print("Initialized a faiss index class.")

  def create_index_for_embedded_spectra(self, database_ids_file, ids_embedded_spectra_path, output_path):
    """

        :param database_ids_file:
        :param ids_embedded_spectra_path:
        :param output_path:
        :return:
        """

    raw_ids, embedded_file_list = [], []
    if os.path.exists(database_ids_file):
      database_ids = np.load(database_ids_file).tolist()
    else:
      database_ids = []
      print("\"" + database_ids_file + " \" does not exist, it will be created!")

    index = self.make_faiss_index_ivf64()

    if str(ids_embedded_spectra_path).endswith("/"):
      dir_path = ids_embedded_spectra_path
    else:
      dir_path = ids_embedded_spectra_path + "/"

    embedded_spectra_file_list = os.listdir(ids_embedded_spectra_path)
    for i in range(len(embedded_spectra_file_list)):
      if embedded_spectra_file_list[i].endswith("_embedded.txt"):
        embedded_file_list.append(embedded_spectra_file_list[i])

    for j in range(len(embedded_file_list)):
      embedded_spectra_data = pd.read_csv(dir_path + embedded_file_list[j], sep="\t", index_col=None)
      ids_data = embedded_spectra_data["ids"].values
      spectra_vectors = embedded_spectra_data["embedded_spectra"].values
      tmp_data = []
      for vec in spectra_vectors:
        tmp_data.append(ast.literal_eval(vec))
      tmp_spectra_vectors = np.vstack(tmp_data)
      tmp_data.clear()

      # Self checking
      self_update_id_bool = False
      if len(ids_data.tolist()) != len(set(ids_data.tolist())):
        self_update_new_ids = []
        self_raw_ids_dict = dict.fromkeys(ids_data.tolist())
        for self_new_id in ids_data:
          self_tmp_id = self_new_id
          while self_raw_ids_dict.keys().__contains__(self_tmp_id):
            self_tmp_id += 1
          if self_tmp_id != self_new_id:
            self_update_id_bool = True
          self_update_new_ids.append(self_tmp_id)
        print("Need to self update ids? {}".format(self_update_id_bool))

        # Check with database_ids
        final_ids, update_id_bool = self.check_ids_with_database(database_ids, self_update_new_ids)

      else:
        # Check with database_ids
        final_ids, update_id_bool = self.check_ids_with_database(database_ids, ids_data.tolist())

      if update_id_bool is True or self_update_id_bool is True:
        update_ids_df = pd.DataFrame({"ids": final_ids})
        ids_vstack_df = pd.concat(
          [update_ids_df, embedded_spectra_data["usi"], embedded_spectra_data["embedded_spectra"]], axis=1)
        store_embed_new_file = dir_path + str(embedded_file_list[j]).strip('embedded.txt') + 'new_ids_embedded.txt'
        ids_vstack_df.to_csv(store_embed_new_file, sep="\t", header=True, index=None,
                             columns=["ids", "usi", "embedded_spectra"])
        print("Update ids for " + str(embedded_file_list[j]) + ", and save in new file:" + store_embed_new_file)

      # index train and add_with_ids
      index.train(tmp_spectra_vectors.astype('float32'))
      index.add_with_ids(tmp_spectra_vectors.astype('float32'), np.array(final_ids))
      raw_ids.extend(final_ids)
      database_ids.extend(final_ids)

    ids_save_file = output_path.strip('.index') + '_ids.npy'
    np.save(database_ids_file, database_ids)
    print("Wrote all database ids to {}".format(database_ids_file))
    np.save(ids_save_file, raw_ids)
    print("Wrote FAISS index ids to {}".format(ids_save_file))
    self.write_faiss_index(index, output_path)

  def merge_indexes(self, input_indexes, output):
    """

        :param input_indexes:
        :param output:
        :return:
        """

    all_ids = []
    index = None
    for input_index in input_indexes:
      print(input_index)
      dirname, filename = os.path.split(os.path.abspath(input_index))

      # ids
      # ids_file = input_index.strip(".index")+ "_ids.npy"
      ids_file = dirname + "/" + filename.strip(".index") + "ids.npy"
      ids_data = np.load(ids_file).tolist()
      all_ids.extend(ids_data)

      # index
      input_index_data = faiss.read_index(input_index)
      if not index:
        index = input_index_data
      else:
        num = len(ids_data)
        index.merge_from(input_index_data, num)

    # Wrote to output file
    # output_path, output_file = os.path.split(os.path.abspath(output))
    ids_save_file = output.strip('.index') + '_ids.npy'
    # ids_save_file = output_path + "/" + output.strip('.index') + '_ids.npy'
    np.save(ids_save_file, all_ids)
    print("Wrote FAISS index database ids to {}".format(ids_save_file))
    self.write_faiss_index(index, output)

  def check_ids_with_database(self, database_ids, self_update_new_ids):

    # Check with database_ids
    update_id_bool = False
    final_ids, update_new_ids = [], []
    if len(database_ids) != 0:
      raw_ids_dict = dict.fromkeys(database_ids)
      for new_id in self_update_new_ids:
        tmp_id = new_id
        while raw_ids_dict.keys().__contains__(tmp_id):
          tmp_id += 1
        if tmp_id != new_id:
          update_id_bool = True
        update_new_ids.append(tmp_id)
      final_ids = update_new_ids

      print("Need to update ids? {}".format(update_id_bool))
    else:
      final_ids = self_update_new_ids

    return final_ids, update_id_bool

  def make_faiss_index_flat(self, n_dimensions, index_type='ivfflat'):
    """
        Make a fairly general-purpose FAISS index
        :param n_dimensions:
        :param index_type: Type of index to build: flat or ivfflat. ivfflat is much faster.
        :return:
        """
    print("Making index of type {}".format(index_type))
    # if faiss.get_num_gpus():
    #     gpu_resources = faiss.StandardGpuResources()
    #     if index_type == 'flat':
    #         config = faiss.GpuIndexFlatConfig()
    #         index = faiss.GpuIndexFlatL2(gpu_resources, n_dimensions, config)
    #     elif index_type == 'ivfflat':
    #         config = faiss.GpuIndexIVFFlatConfig()
    #         index = faiss.GpuIndexIVFFlat(gpu_resources, n_dimensions, DEFAULT_IVF_NLIST, faiss.METRIC_L2, config)
    #     else:
    #         raise ValueError("Unknown index_type %s" % index_type)
    # else:
    print("Using CPU.")
    if index_type == 'flat':
      index = faiss.IndexFlatL2(n_dimensions)
    elif index_type == 'ivfflat':
      quantizer = faiss.IndexFlatL2(n_dimensions)
      index = faiss.IndexIVFFlat(quantizer, n_dimensions, DEFAULT_IVF_NLIST, faiss.METRIC_L2)
    else:
      raise ValueError("Unknown index_type %s" % index_type)
    return index

  def make_faiss_index_idmap(self, n_dimensions):
    """
        Make a fairly general-purpose FAISS index
        :param n_dimensions:
        :return:
        """
    print("Making index ...")
    tmp_index = faiss.IndexFlatL2(n_dimensions)
    index = faiss.IndexIDMap(tmp_index)
    return index

  def make_faiss_index_ivf64(self):
    """
        Save a FAISS index. If we're on GPU, have to convert to CPU index first
        :return:
        """
    index = faiss.index_factory(32, "IVF64,Flat")
    return index

  def write_faiss_index(self, index, out_filepath):
    """
        Save a FAISS index. If we're on GPU, have to convert to CPU index first
        :param out_filepath:
        :param index:
        :return:
        """
    # if faiss.get_num_gpus():
    #     print("Converting index from GPU to CPU...")
    #     index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, out_filepath)
    print("Wrote FAISS index to {}".format(out_filepath))

  def read_faiss_index_gpu(self, index_filepath):
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
