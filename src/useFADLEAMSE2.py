# -*- coding:utf-8 -*-
"""
This is a spectra encoding and embeding program!
Create by qincy, April 17,2019
"""

import argparse
import os

import torch
from pyteomics.mgf import read

from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from numpy import concatenate
from numba import njit

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

    def forward_once(self, preInfo, fragInfo, refSpecInfo):
        preInfo = self.fc1_1(preInfo)
        preInfo = F.selu(preInfo)
        preInfo = self.fc1_2(preInfo)
        preInfo = F.selu(preInfo)
        preInfo = preInfo.view(preInfo.size(0), -1)

        fragInfo = self.cnn21(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = self.maxpool21(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = self.cnn22(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = self.maxpool22(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = fragInfo.view(fragInfo.size(0), -1)

        refSpecInfo = self.cnn11(refSpecInfo)
        refSpecInfo = F.selu(refSpecInfo)
        refSpecInfo = self.maxpool11(refSpecInfo)
        refSpecInfo = F.selu(refSpecInfo)
        refSpecInfo = refSpecInfo.view(refSpecInfo.size(0), -1)

        output = torch.cat((preInfo, fragInfo, refSpecInfo), 1)
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

class EncodeDataset():

    def __init__(self, mgf_file, ref_spectra, miss_saveName):
        if not os.path.exists(mgf_file):
            raise RuntimeError("Can not find mgf file: '%s'" % mgf_file)

        self.MGF = read(mgf_file, convert_arrays=1)
        self.len = 0
        for da in self.MGF:
            self.len += 1

        self.data = self.transform(mgf_file, ref_spectra, miss_saveName)

        # print(self.MGF.__iter__())
        # self.len = len(self.MGF.__iter__())

    def transform(self, mgf_file, ref_spectra, miss_saveName):
        self.mgf_dataset = None
        print('Start to calculate data set...')
        #五百个参考的谱图
        # reference_spectra = read("./0722_500_rf_spectra.mgf", convert_arrays=1)
        reference_spectra = read(ref_spectra, convert_arrays=1)
        reference_intensity = np.array([bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in reference_spectra])

        # 先将500个参考谱图的点积结果计算出来
        ndp_r_spec_list = caculate_r_spec(reference_intensity)

        peakslist1, precursor_feature_list1 = [], []
        ndp_spec_list = []
        i, j, k = 0, 0, 0
        charge_none_record, charge_none_list = 0, []
        for s1 in read(mgf_file, convert_arrays=1):
            print(s1)
            if s1.get('params').get('charge').__str__()[0] == "N":
                charge_none_record += 1
                spectrum_id = s1.get('params').get('title')
                charge_none_list.append(spectrum_id)
                continue
            else:
                print("new")
                charge1 = int(s1.get('params').get('charge').__str__()[0])

            bin_s1 = bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
            ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
            peakslist1.append(bin_s1)
            ndp_spec_list.append(ndp_spec1)
            mass1 = float(s1.get('params').get('pepmass')[0])

            precursor_feature1 = np.concatenate((self.gray_code(mass1), self.charge_to_one_hot(charge1)))
            precursor_feature_list1.append(precursor_feature1)

            if len(peakslist1) == 500:
                i += 1
                tmp_precursor_feature_list1 = np.array(precursor_feature_list1)
                intensList01 = np.array(peakslist1)

                # 归一化点积的计算
                tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1), np.array(ndp_spec_list))

                tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
                spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

                if i == 1:
                    self.mgf_dataset = spectrum01
                else:
                    self.mgf_dataset = np.vstack((self.mgf_dataset, spectrum01))

                peakslist1.clear()
                precursor_feature_list1.clear()
                ndp_spec_list.clear()

                j = i * 500

            elif (j+500+charge_none_record) > self.len:
                num = self.len - j - charge_none_record
                k += 1
                if num == k:

                    tmp_precursor_feature_list1 = np.array(precursor_feature_list1)

                    intensList01 = np.array(peakslist1)

                    # 归一化点积的计算
                    tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list,
                                                                  np.array(peakslist1), np.array(ndp_spec_list))

                    tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
                    spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

                    self.mgf_dataset = np.vstack((self.mgf_dataset, spectrum01))

                    peakslist1.clear()
                    precursor_feature_list1.clear()
                    ndp_spec_list.clear()
                else:
                    continue
            # else:
            #     continue

        np_mr = np.array(charge_none_list)
        df_mr = pd.DataFrame(np_mr, index=None, columns=None)
        df_mr.to_csv(miss_saveName)
        del self.MGF
        del charge_none_list
        print("Charge Missing Number:{}".format(charge_none_record))
        return self.mgf_dataset

    def getData(self):
        return self.data

    def gray_code(self, number):
        """
        to get the gray code:\n
            1. a = get the num's binary form
            2. b = shift a one bit from left to right, put zero at the left position
            3. gray code = a xor b
            bin(num ^ (num >> 1))
        :param number:
        :return:np.array  gray code array for num
        """
        number = np.int(number)
        bit = 27
        shift = 1
        gray_code = np.binary_repr(np.bitwise_xor(number, np.right_shift(number, shift)), bit)
        return np.asarray(' '.join(gray_code).split(), dtype=float)

    def charge_to_one_hot(self, c: int):
        """
        encode charge with one-hot format for 1-7
        :param c:
        :return:
        """
        maximum_charge = 7
        charge = np.zeros(maximum_charge, dtype=float)
        if c > maximum_charge: c = maximum_charge
        charge[c - 1] = c
        return charge

@njit
def caculate_spec(bin_spec):
    ndp_spec1 = np.math.sqrt(np.dot(bin_spec, bin_spec))
    return ndp_spec1

@njit
def caculate_r_spec(reference_intensity):
    ndp_r_spec_list = np.zeros(500)
    for x in range(500):
        ndp_r_spec = np.math.sqrt(np.dot(reference_intensity[x], reference_intensity[x]))
        ndp_r_spec_list[x] = ndp_r_spec
    return ndp_r_spec_list

@njit
def get_bin_index(mz, min_mz, bin_size):
    relative_mz = mz - min_mz
    return max(0, int(np.floor(relative_mz / bin_size)))

@njit
def bin_spectrum(mz_array, intensity_array, max_mz=2500, min_mz=50.5, bin_size=1.0005079):
    """
    bin spectrum and this algorithm reference from 'https://github.com/dhmay/param-medic/blob/master/parammedic/binning.pyx'
    :param mz_array:
    :param intensity_array:
    :param max_mz:
    :param min_mz:
    :param bin_size:
    :return:
    """
    # key = mz_array.__str__()
    # if key in spectrum_dict.keys():  # use cache just take 4s
    #     # if False: use the old one may take 7s for 50
    #     return spectrum_dict[key]
    # else:
    nbins = int(float(max_mz - min_mz) / float(bin_size)) + 1
    results = np.zeros(nbins)

    for index in range(len(mz_array)):
        mz = mz_array[index]
        intensity = intensity_array[index]
        intensity = np.math.sqrt(intensity)
        if mz < min_mz or mz > max_mz:
            continue
        bin_index = get_bin_index(mz, min_mz, bin_size)

        if bin_index < 0 or bin_index > nbins - 1:
            continue
        if results[bin_index] == 0:
            results[bin_index] = intensity
        else:
            results[bin_index] += intensity

    intensity_sum = results.sum()

    if intensity_sum > 0:
        results /= intensity_sum
        # spectrum_dict[key] = results
    else:
        print('zero intensity found')
    return results

@njit
def caculate_nornalization_dp(reference, ndp_r_spec_list, bin_spectra, ndp_bin_sp):
    ndp_r_spec_list = ndp_r_spec_list.reshape(ndp_r_spec_list.shape[0],1)
    ndp_bin_sp = ndp_bin_sp.reshape(ndp_bin_sp.shape[0], 1)
    tmp_dp_list = np.dot(bin_spectra, np.transpose(reference))
    dvi = np.dot(ndp_bin_sp, np.transpose(ndp_r_spec_list))
    result = tmp_dp_list / dvi
    return result

class LoadDataset(data.dataset.Dataset):
    def __init__(self, data):
        self.dataset = data

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return self.dataset.shape[0]

class EmbedDataset():
    def __init__(self):
        self.out_list = []

    def embedding_dataset(self, model, mgfFile, ref_spectra, storeEmbedFile, saveName):

        # for gpu
        # batch = 1000
        # net = torch.load(model)

        # for cpu
        batch = 1
        net = torch.load(model, map_location='cpu')

        print("Start encoding all spectra ...")
        vstack_data = EncodeDataset(mgfFile, ref_spectra, saveName).getData()
        dataset = LoadDataset(vstack_data)
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=1)

        print("Start to embed all spectra ... ")
        for j, test_data in enumerate(dataloader, 0):

            spectrum01 = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])

            input1_1 = spectrum01[:, :, :500]
            input1_2 = spectrum01[:, :, 500:2949]
            input1_3 = spectrum01[:, :, 2949:]

            # for gpu
            # refSpecInfo1, fragInfo1, preInfo1 = input1_3.cuda(), input1_2.cuda(), input1_1.cuda()
            # output01 = net.forward_once(refSpecInfo1, fragInfo1, preInfo1)
            # out1 = output01.cpu().detach().numpy()

            # for cpu
            output01 = net.forward_once(input1_3, input1_2, input1_1)
            out1 = output01.detach().numpy()[0]

            if j == 0:
                self.out_list = out1
            else:
                self.out_list = np.vstack((self.out_list, out1))

        np.savetxt(storeEmbedFile, self.out_list)

def executeEmbedding(model, input_mgf_file, ref_spectra, output_embedded_file):

    charge_miss_sid = input_mgf_file + "_charge-missing.info"

    embedder01 = EmbedDataset()
    embedder01.embedding_dataset(model, input_mgf_file, ref_spectra, output_embedded_file, charge_miss_sid)

def declare_gather_args():
    """
    Declare all arguments, parse them, and return the args dict.
    Does no validation beyond the implicit validation done by argparse.
    return: a dict mapping arg names to values
    """

    # declare args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=argparse.FileType('r'), help='input model file')
    parser.add_argument('--input', type=argparse.FileType('r'), required=True, help='input mgf file')
    parser.add_argument('--ref_spectra', type=argparse.FileType('r'), help='input ref. spectra file', default="./siamese_modle_reference/0722_500_rf_spectra.mgf")
    parser.add_argument('--output', type=argparse.FileType('w'), required=True, help='output vectors file')
    return parser.parse_args()

if __name__ == '__main__':

    #parameters
    # python useFASLEAMSE.py ../siamese_modle_reference/080802_20_1000_NM500R_model.pkl --input ./data/130402_08.mgf --output ./data/test.csv
    
    args = declare_gather_args()
    model = args.model.name
    input_file = args.input.name
    ref_spectra_file = args.ref_spectra.name
    output_file = args.output.name

    executeEmbedding(model, input_file, ref_spectra_file, output_file)
