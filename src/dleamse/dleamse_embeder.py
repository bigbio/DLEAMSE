# -*- coding:utf-8 -*-

"""
Embed spectra.

"""
from dleamse.dleamse_encoder import encode_spectra

import torch

from torch.utils import data
import torch.nn.functional as func
import torch.nn as nn
import numpy as np
import pandas as pd


class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()

        self.fc1_1 = nn.Linear(34, 32)
        self.fc1_2 = nn.Linear(32, 5)

        self.cnn11 = nn.Conv1d(1, 30, 3)
        self.maxpool11 = nn.MaxPool1d(2)

        self.cnn21 = nn.Conv1d(1, 30, 3)
        self.maxpool21 = nn.MaxPool1d(2)
        self.cnn22 = nn.Conv1d(30, 30, 3)
        self.maxpool22 = nn.MaxPool1d(2)

        self.fc2 = nn.Linear(25775, 32)

    def forward_once(self, pre_info, frag_info, ref_spec_info):
        pre_info = self.fc1_1(pre_info)
        pre_info = func.selu(pre_info)
        pre_info = self.fc1_2(pre_info)
        pre_info = func.selu(pre_info)
        pre_info = pre_info.view(pre_info.size(0), -1)

        frag_info = self.cnn21(frag_info)
        frag_info = func.selu(frag_info)
        frag_info = self.maxpool21(frag_info)
        frag_info = func.selu(frag_info)
        frag_info = self.cnn22(frag_info)
        frag_info = func.selu(frag_info)
        frag_info = self.maxpool22(frag_info)
        frag_info = func.selu(frag_info)
        frag_info = frag_info.view(frag_info.size(0), -1)

        ref_spec_info = self.cnn11(ref_spec_info)
        ref_spec_info = func.selu(ref_spec_info)
        ref_spec_info = self.maxpool11(ref_spec_info)
        ref_spec_info = func.selu(ref_spec_info)
        ref_spec_info = ref_spec_info.view(ref_spec_info.size(0), -1)

        output = torch.cat((pre_info, frag_info, ref_spec_info), 1)
        output = self.fc2(output)
        return output

    def forward(self, spectrum01, spectrum02):
        spectrum01 = spectrum01.reshape(spectrum01.shape[0], 1, spectrum01.shape[1])
        spectrum02 = spectrum02.reshape(spectrum02.shape[0], 1, spectrum02.shape[1])

        input1_1 = spectrum01[:, :, :500]
        input1_2 = spectrum01[:, :, 500:2949]
        input1_3 = spectrum01[:, :, 2949:]

        input2_1 = spectrum02[:, :, :500]
        input2_2 = spectrum02[:, :, 500:2949]
        input2_3 = spectrum02[:, :, 2949:]

        refSpecInfo1, fragInfo1, preInfo1 = input1_3.cuda(), input1_2.cuda(), input1_1.cuda()
        refSpecInfo2, fragInfo2, preInfo2 = input2_3.cuda(), input2_2.cuda(), input2_1.cuda()

        output01 = self.forward_once(refSpecInfo1, fragInfo1, preInfo1)
        output02 = self.forward_once(refSpecInfo2, fragInfo2, preInfo2)

        return output01, output02


class LoadDataset(data.dataset.Dataset):
    def __init__(self, data):
        self.dataset = data

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return self.dataset.shape[0]


class EmbedDataset:
    def __init__(self, model, ids_data, vstack_encoded_spectra, store_embed_file, use_gpu):
        self.ids_vstack_df = None
        self.out_list = []
        self.embedding_dataset(model, ids_data, vstack_encoded_spectra, store_embed_file, use_gpu)

    def get_data(self):
        return self.ids_vstack_df

    def embedding_dataset(self, model, ids_data, encoded_spectra_data, store_embed_file, use_gpu):

        if use_gpu is True:
            # for gpu
            batch = 1000
            net = torch.load(model)
        else:
            # for cpu
            batch = 1
            net = torch.load(model, map_location='cpu')


        dataset = LoadDataset(encoded_spectra_data)

        dataloader = data.DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=1)

        print("Start spectra embedding ... ")
        for j, test_data in enumerate(dataloader, 0):

            spectrum01 = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])

            input1_1 = spectrum01[:, :, :500]
            input1_2 = spectrum01[:, :, 500:2949]
            input1_3 = spectrum01[:, :, 2949:]

            if use_gpu is True:
                # for gpu
                refSpecInfo1, fragInfo1, preInfo1 = input1_3.cuda(), input1_2.cuda(), input1_1.cuda()
                output01 = net.forward_once(refSpecInfo1, fragInfo1, preInfo1)
                out1 = output01.cpu().detach().numpy()
            else:
                # for cpu
                output01 = net.forward_once(input1_3, input1_2, input1_1)
                out1 = output01.detach().numpy()[0]

            if j == 0:
                self.out_list = out1
            else:
                self.out_list = np.vstack((self.out_list, out1))

        vstack_data_df = pd.DataFrame({"embedded_spectra": self.out_list.tolist()})
        self.ids_vstack_df = pd.concat([ids_data, vstack_data_df], axis=1)
        self.ids_vstack_df.to_csv(store_embed_file, header=True, index=None, columns=["ids", "embedded_spectra"])


def embed_spectra(model, ids_usi_data, vstack_encoded_spectra, output_embedd_file, **kwargs):
    """
    Embed spectra
    :param model:  .pkl format embedding model
    :param ids_usi_data: ids-usi dataframe data
    :param vstack_encoded_spectra: encoded spectra file for embedding
    :param kwargs: bool, default=False;
    :return: ids-embedded_spectra data
    """

    if kwargs.keys().__contains__("use_gpu"):
        use_gpu = kwargs["use_gpu"]
    else:
        use_gpu = False

    ids_data = ids_usi_data["ids"]

    ids_embedded_spectra = EmbedDataset(model, ids_data, vstack_encoded_spectra, output_embedd_file, use_gpu).get_data()

    print("Finish spectra embedding, save embedded spectra to " + output_embedd_file + "!")
    return ids_embedded_spectra
