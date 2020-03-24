# -*- coding:utf-8 -*-


import numpy as np
import faiss
import ast

DEFAULT_IVF_NLIST = 100

class FaissWriteIndex:

    def __init__(self):
        self.tmp = None
        print("Initialized a faiss index class.")

    def create_index(self, ids_embedded_spectra, ids_save_file, output_path):
        """
        Create faiss indexIDMap index
        :param spectra_vectors: spectra embedded data
        :param usi_data: coresponding usi data
        :param output_path: output file path
        :return:
        """
        ids_data = ids_embedded_spectra["ids"].values
        spectra_vectors = ids_embedded_spectra["embedded_spectra"].values
        if type(spectra_vectors[0]) == type("test"):
            tmp_data = []
            for vec in spectra_vectors:
                tmp_data.append(ast.literal_eval(vec))
            tmp_spectra_vectors = np.vstack(tmp_data)
            tmp_data.clear()
        else:
            tmp_spectra_vectors = np.vstack(spectra_vectors)
        np.save(ids_save_file, ids_data)
        n_embedded_dim = tmp_spectra_vectors.shape[1]
        index = self.make_faiss_index_IDMap(n_embedded_dim)
        index.add_with_ids(tmp_spectra_vectors.astype('float32'), ids_data)
        self.write_faiss_index(index, output_path)

    def add_index(self, raw_index, raw_ids_file, new_ids_embedded_data, output_index_ids_file, output_index_file):
        """
        Add new_index data to a raw_index
        :param raw_index: Raw index file
        :param new_index: New index file
        :param new_usi_data: New index's corresponding usi data
        :param output_path: Output file path
        :return:
        """
        new_ids_data = new_ids_embedded_data["ids"].values.tolist()
        new_embedded_spectra = new_ids_embedded_data["embedded_spectra"]

        #Determine whether the new index is the same as the original index
        raw_ids_dict = dict.fromkeys(np.load(raw_ids_file).tolist(), [])
        update_new_ids = []
        update_id_bool = False
        for new_id in new_ids_data:
            print("new id:{}".format(new_id))
            tmp_id = new_id
            while raw_ids_dict.keys().__contains__(tmp_id):
                tmp_id += 1
            if tmp_id != new_id:
                update_id_bool = True
            update_new_ids.append(tmp_id)

        if update_id_bool is True:
            print("Need to update new sepctra list's ids, save updated ids to "+str(output_index_ids_file).strip(".npy") + "_new_spectra_updated_ids.npy.")
            np.save(str(output_index_ids_file).strip(".npy") + "_new_spectra_updated_ids.npy.", update_new_ids)
            # add index
            raw_index.add_with_ids(new_embedded_spectra.astype('float32'), np.array(update_new_ids))
            new_faiss_index_ids = np.load(raw_ids_file).tolist().extend(update_new_ids)
            np.save(output_index_ids_file, new_faiss_index_ids)
            self.write_faiss_index(raw_index, output_index_file)
        else:
            update_new_ids.clear()
            # add index
            raw_index.add_with_ids(new_embedded_spectra.astype('float32'), np.array(new_ids_data))
            new_faiss_index_ids = np.load(raw_ids_file).tolist().extend(new_ids_data)
            np.save(output_index_ids_file, new_faiss_index_ids)
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
        print("Making index ： IDMap...")
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