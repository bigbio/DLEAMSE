# -*- coding:utf-8 -*-
import os
import sys

import numpy as np
import pandas as pd
import faiss
import ast

DEFAULT_IVF_NLIST = 100

class FaissWriteIndex:

    def __init__(self):
        self.tmp = None
        print("Initialized a faiss index class.")

    def create_index_for_embedded_spectra(self, ids_embedded_spectra_path, ids_save_file, output_path):
        """
        Create faiss indexIDMap index
        :param spectra_vectors: spectra embedded data
        :param usi_data: coresponding usi data
        :param output_path: output file path
        :return:
        """

        index = self.make_faiss_index_IDMap(32)
        raw_ids = []

        embedded_spectra_file_list = os.listdir(ids_embedded_spectra_path)

        if str(ids_embedded_spectra_path).endswith("/"):
            dir_path = ids_embedded_spectra_path
        else:
            dir_path = ids_embedded_spectra_path + "/"

        embedded_file_list = []
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

            if j == 0:
                print(j)
                index.add_with_ids(tmp_spectra_vectors.astype('float32'), ids_data)
                raw_ids = ids_data.tolist()

            else:
                # Determine whether the new index is the same as the original index
                raw_ids_dict = dict.fromkeys(raw_ids)
                update_new_ids = []
                update_id_bool = False
                for new_id in ids_data:
                    tmp_id = new_id
                    while raw_ids_dict.keys().__contains__(tmp_id):
                        tmp_id += 1
                    if tmp_id != new_id:
                        update_id_bool = True
                    update_new_ids.append(tmp_id)

                print("Need to update ids? {}".format(update_id_bool))
                if update_id_bool is True:
                    update_ids_df = pd.DataFrame({"ids":update_new_ids})
                    ids_vstack_df = pd.concat([update_ids_df, embedded_spectra_data["embedded_spectra"]], axis=1)
                    store_embed_new_file = dir_path + str(embedded_file_list[j]).strip("_embedded.txt")+ "_new_ids_embedded.txt"
                    ids_vstack_df.to_csv(store_embed_new_file, sep="\t", header=True, index=None, columns=["ids", "embedded_spectra"])
                    print("Update ids for " + str(embedded_file_list[j]) + ", and save in new file:" + store_embed_new_file )

                    # add index
                    index.add_with_ids(tmp_spectra_vectors.astype('float32'), np.array(update_new_ids))
                    raw_ids.extend(update_new_ids)
                else:
                    index.add_with_ids(tmp_spectra_vectors.astype('float32'), ids_data)
                    raw_ids.extend(ids_data.tolist())

        np.save(ids_save_file, raw_ids)
        self.write_faiss_index(index, output_path)

    def add_embedded_spectra_to_index(self, raw_index, raw_ids_file, new_embedded_spectra_path, output_index_ids_file, output_index_file):
        """
        Add new_index data to a raw_index
        :param raw_index: Raw index file
        :param new_index: New index file
        :param new_usi_data: New index's corresponding usi data
        :param output_path: Output file path
        :return:
        """
        raw_index = faiss.read_index(raw_index)

        raw_ids = np.load(raw_ids_file).tolist()

        embedded_spectra_file_list = os.listdir(new_embedded_spectra_path)
        if str(new_embedded_spectra_path).endswith("/"):
            dir_path = new_embedded_spectra_path
        else:
            dir_path = new_embedded_spectra_path + "/"

        embedded_file_list = []
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

            # Determine whether the new index is the same as the original index
            raw_ids_dict = dict.fromkeys(raw_ids)
            update_new_ids = []
            update_id_bool = False
            for new_id in ids_data:
                tmp_id = new_id
                while raw_ids_dict.keys().__contains__(tmp_id):
                    tmp_id += 1
                if tmp_id != new_id:
                    update_id_bool = True
                update_new_ids.append(tmp_id)

            print("Need to update ids? {}".format(update_id_bool))
            if update_id_bool is True:
                update_ids_df = pd.DataFrame({"ids": update_new_ids})
                ids_vstack_df = pd.concat([update_ids_df, embedded_spectra_data["embedded_spectra"]], axis=1)
                store_embed_new_file = dir_path + str(embedded_file_list[j]).strip(
                    "_embedded.txt") + "_new_ids_embedded.txt"
                ids_vstack_df.to_csv(store_embed_new_file, sep="\t", header=True, index=None,
                                     columns=["ids", "embedded_spectra"])
                print("Update ids for " + str(
                    embedded_file_list[j]) + ", and save in new file:" + store_embed_new_file)

                # add index
                raw_index.add_with_ids(tmp_spectra_vectors.astype('float32'), np.array(update_new_ids))
                raw_ids.extend(update_new_ids)
            else:
                raw_index.add_with_ids(tmp_spectra_vectors.astype('float32'), ids_data)
                raw_ids.extend(ids_data.tolist())

        np.save(output_index_ids_file, raw_ids)
        self.write_faiss_index(raw_index, output_index_file)

    def make_faiss_indexFlat(self, n_dimensions, index_type='flat'):
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

    def make_faiss_index_IDMap(self, n_dimensions):
        """
        Make a fairly general-purpose FAISS index
        :param n_dimensions:
        :return:
        """
        print("Making index ...")
        tmp_index = faiss.IndexFlatL2(n_dimensions)
        index = faiss.IndexIDMap(tmp_index)
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

