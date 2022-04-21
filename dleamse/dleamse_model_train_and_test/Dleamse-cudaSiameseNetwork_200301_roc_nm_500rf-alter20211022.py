# -*- coding:utf-8 -*-
#---------------------------------
# This is a Demo of SiameseNetwork
# created by pytorch.
# Use a fixed 500 reference spectrum
# Do an overall dot product calculation for a large table
#---------------------------------
import math
import os
import logging
import time

from pyteomics.mgf import read
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from torch.utils import data
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from numpy import concatenate
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegressionCV
import matplotlib as mpl



def show_plot(batch_size, acc, loss, saveName):
    # plt.plot(iteration, loss)
    # plt.show()
    plt.figure(figsize=(16, 9))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(acc)), acc)
    plt.title('Acc for Epoch: %s Batch_size: %s' % (len(acc), batch_size))
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(loss)), loss, 'r')
    plt.title('Loss for Iteration: %s Batch_size: %s' % (len(loss), batch_size))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('../Picture/' + saveName + '.jpg')
    # plt.show()
    plt.close()


# # Sort the input spectrogram and spectrogram label, and synthesize the encoded training data
# class NewMGFDataSet(data.dataset.Dataset):

#     def __init__(self, mgf_file, csv_file, reference_spectra_number=500):
#         if not os.path.exists(mgf_file):
#             raise RuntimeError("Can not find mgf file: '%s'" % mgf_file)
#         if not os.path.exists(csv_file):
#             raise RuntimeError("Can not find csv file: '%s'" % csv_file)
#         self.MGF = {}
#         self.mgf_dataset = []
#         self.load_file(mgf_file, csv_file)
#         self.transform(reference_spectra_number)
#         # print(self.mgf_dataset)

#     def load_file(self, mgf_path, csv_path):
#         print('Start to load file data...')
#         info = pd.read_csv(csv_path, header=None)
#         self.spectrum1 = info[0].tolist()
#         self.spectrum2 = info[1].tolist()
#         self.label = info[2].tolist()
#         for mgf in read(mgf_path, convert_arrays=1):
#             self.MGF[mgf.get('params').get('title').replace('id=', '')] = mgf
#         print('Finish to load data...')

#     def transform(self, reference_spectra_number):
#         print('Start to calculate data set...')

#         global spectrum_dict
#         spectrum_dict = {}
#         # Five hundred reference spectra
#         rfDataFrame = pd.read_csv("../Data/500RfSpectraBinData.csv", header=None, index_col=None)
#         reference_intensity = rfDataFrame.values

#         peakslist1, precursor_feature_list1 = [], []
#         peakslist2, precursor_feature_list2 = [], []
#         for s1, s2, l in zip(self.spectrum1, self.spectrum2, self.label):
#             s1 = self.MGF[s1]
#             s2 = self.MGF[s2]
#             bin_s1 = self.bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
#             peakslist1.append(bin_s1)

#             mass1 = float(s1.get('params').get('pepmass')[0])
#             charge1 = int(s1.get('params').get('charge').__str__()[0])
#             mz1 = mass1 / charge1
#             precursor_feature1 = np.concatenate((self.gray_code(mass1), self.gray_code(mz1), self.charge_to_one_hot(charge1)))
#             precursor_feature_list1.append(precursor_feature1)

#             bin_s2 = self.bin_spectrum(s2.get('m/z array'), s2.get('intensity array'))
#             peakslist2.append(bin_s2)

#             mass2 = float(s2.get('params').get('pepmass')[0])
#             charge2 = int(s2.get('params').get('charge').__str__()[0])
#             mz2 = mass2 / charge2
#             precursor_feature2 = np.concatenate((self.gray_code(mass2), self.gray_code(mz2), self.charge_to_one_hot(charge2)))
#             precursor_feature_list2.append(precursor_feature2)

#         intensList01 = np.array(peakslist1)
#         intensList02 = np.array(peakslist2)
#         refMatrix = np.transpose(reference_intensity)

#         # j, k = 0, 0
#         # num = math.ceil(len(intensList01) / 10000) 
#         # tmp_dplist01, tmp_dplist01 = [], []
#         # for i in range(num):
#         #     DPList01, DPList02 = [], []
#         #     j = i * 10000
#         #     if j > len(intensList01):
#         #         j = len(intensList01)
#         #         DPList01 = np.dot(intensList01[k:j, :], refMatrix)
#         #         DPList02 = np.dot(intensList02[k:j, :], refMatrix)
#         #     DPList01 = np.dot(intensList01[k:j, :], refMatrix)
#         #     DPList02 = np.dot(intensList02[k:j, :], refMatrix)
#         #     k = j
#         #     tmp_dplist01 = concatenate((tmp_dplist01, DPList01), axis=0)
#         #     tmp_dplist02 = concatenate((tmp_dplist02, DPList02), axis=0)

#         DPList01 = np.dot(intensList01, refMatrix)
#         DPList02 = np.dot(intensList02, refMatrix)

#         precursor_feature_list1 = np.array(precursor_feature_list1)
#         precursor_feature_list2 = np.array(precursor_feature_list2)

#         label = np.array(self.label)

#         tmp01 = concatenate((DPList01, intensList01), axis=1)
#         tmp02 = concatenate((DPList02, intensList02), axis=1)
#         spectrum01 = concatenate((tmp01, precursor_feature_list1), axis=1)
#         spectrum02 = concatenate((tmp02, precursor_feature_list2), axis=1)

#         label = np.array(label.reshape(label.shape[0], 1))
#         tmp_data = concatenate((spectrum01, spectrum02), axis=1)
#         self.mgf_dataset = concatenate((tmp_data, label), axis=1)
#         print("self.mgf_dataset 维度:",self.mgf_dataset.shape)

#         del self.MGF
#         print('Finish to calculate data set...')

#     def __getitem__(self, item):
#         return self.mgf_dataset[item]
#     def __len__(self):
#         return len(self.mgf_dataset)

#     def gray_code(self, number):
#         """
#         to get the gray code:\n
#             1. a = get the num's binary form
#             2. b = shift a one bit from left to right, put zero at the left position
#             3. gray code = a xor b
#             bin(num ^ (num >> 1))
#         :param number:
#         :return:np.array  gray code array for num
#         """
#         # assert num.is_integer(), 'Parameter "num" must be integer'
#         number = np.int(number)
#         # we need 27-bit "Gray Code"
#         bit = 27
#         shift = 1
#         gray_code = np.binary_repr(np.bitwise_xor(number, np.right_shift(number, shift)), bit)
#         # print(type(gray_code))
#         return np.asarray(' '.join(gray_code).split(), dtype=float)

#     def charge_to_one_hot(self, c: int):
#         """
#         encode charge with one-hot format for 1-7
#         :param c:
#         :return:
#         """
#         maximum_charge = 7
#         charge = np.zeros(maximum_charge, dtype=float)
#         # if charge bigger than 7, use 7 instead
#         if c > maximum_charge: c = maximum_charge
#         charge[c - 1] = c
#         return charge

#     def get_bin_index(self, mz, min_mz, bin_size):
#         relative_mz = mz - min_mz
#         return max(0, int(np.floor(relative_mz / bin_size)))

#     def bin_spectrum(self, mz_array, intensity_array, max_mz=2500, min_mz=50.5, bin_size=1.0005079):
#         """
#         bin spectrum and this algorithm reference from 'https://github.com/dhmay/param-medic/blob/master/parammedic/binning.pyx'
#         :param mz_array:
#         :param intensity_array:
#         :param max_mz:
#         :param min_mz:
#         :param bin_size:
#         :return:
#         """
#         key = mz_array.__str__()
#         if key in spectrum_dict.keys():  # use cache just take 4s
#             # if False: use the old one may take 7s for 50
#             return spectrum_dict[key]
#         else:
#             nbins = int(float(max_mz - min_mz) / float(bin_size)) + 1
#             results = np.zeros(nbins)
#             for index in range(len(mz_array)):
#                 mz = mz_array[index]
#                 intensity = intensity_array[index]
#                 intensity = np.math.sqrt(intensity)
#                 if mz < min_mz or mz > max_mz:
#                     continue
#                 bin_index = self.get_bin_index(mz, min_mz, bin_size)

#                 if bin_index < 0 or bin_index > nbins - 1:
#                     continue

#                 if results[bin_index] == 0:
#                     results[bin_index] = intensity
#                 else:
#                     results[bin_index] += intensity
#                 # results[bin_index] = intensity
#             intensity_sum = results.sum()
#             if intensity_sum > 0:
#                 results /= intensity_sum
#                 spectrum_dict[key] = results
#             else:
#                 logging.debug('zero intensity found')
#         return results
class RawDataSet():

    def __init__(self, mgf_file, csv_file, saveName):
        start_time = time.perf_counter()
        if not os.path.exists(mgf_file):
            raise RuntimeError("Can not find mgf file: '%s'" % mgf_file)
        if not os.path.exists(csv_file):
            raise RuntimeError("Can not find csv file: '%s'" % csv_file)
        self.MGF = {}
        self.mgf_dataset = []
        self.load_file(mgf_file, csv_file)

        end_time = time.perf_counter()
        print("Time consuming to read files：",end_time - start_time)

        start_time_transform = time.perf_counter()
        self.transform(saveName)
        end_time_transform = time.perf_counter()
        print("Transform overall runtime：",end_time_transform - start_time_transform)

    def load_file(self, mgf_path, csv_path):
        print('Start to load file data...')
        info = pd.read_csv(csv_path, header=None)
        self.pairs_num = info.shape[0]
        self.spectrum1 = info[0].tolist()
        self.spectrum2 = info[1].tolist()
        self.label = info[2].tolist()
        for mgf in read(mgf_path, convert_arrays=1):
            self.MGF[mgf.get('params').get('title').replace('id=', '')] = mgf
        print('Finish to load data...')

    def transform(self, saveName):
        print('Start to calculate data set...')

        global spectrum_dict
        spectrum_dict = {}
        # 500 ref spectra
        # reference_spectra = read("../SpectraPairsData/0715_50_rf_spectra.mgf", convert_arrays=1)
        # reference_spectra = read("../SpectraPairsData/0628_100_rf_spectra.mgf", convert_arrays=1)
        start_time_reference_spectra = time.perf_counter()
        reference_spectra = read("./Data/0722_500_rf_spectra.mgf", convert_arrays=1)
        reference_intensity = np.array([self.bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in reference_spectra])
        # First calculate the dot product results of 500 reference spectra
        # ndp_r_spec_list = np.zeros(50)
        # ndp_r_spec_list = np.zeros(100)
        ndp_r_spec_list = np.zeros(500)

        # for x in range(50):
        # for x in range(100):
        for x in range(500):
            ndp_r_spec = np.math.sqrt(np.dot(reference_intensity[x], reference_intensity[x]))
            ndp_r_spec_list[x] = ndp_r_spec

        peakslist1, precursor_feature_list1 = [], []
        peakslist2, precursor_feature_list2 = [], []
        label_list = []
        ndp_spec_list1, ndp_spec_list2 = [], []

        end_time_reference_spectra = time.perf_counter()
        print("reference_spectra read in overall runtime：",end_time_reference_spectra - start_time_reference_spectra)
        
        start_time_transform_for = time.perf_counter()

        mgf_datasetTempList = []
        for s1, s2, l in zip(self.spectrum1, self.spectrum2, self.label):
            s1 = self.MGF[s1]
            s2 = self.MGF[s2]

            bin_s1 = self.bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
            ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1)) # Calculated as a number
            peakslist1.append(bin_s1)
            ndp_spec_list1.append(ndp_spec1) # （n,）
            mass1 = float(s1.get('params').get('pepmass')[0])
            charge1 = int(s1.get('params').get('charge').__str__()[0])
            precursor_feature1 = np.concatenate((self.gray_code(mass1), self.charge_to_one_hot(charge1)))
            precursor_feature_list1.append(precursor_feature1)

            bin_s2 = self.bin_spectrum(s2.get('m/z array'), s2.get('intensity array'))
            ndp_spec2 = np.math.sqrt(np.dot(bin_s2, bin_s2))
            peakslist2.append(bin_s2)
            ndp_spec_list2.append(ndp_spec2)
            mass2 = float(s2.get('params').get('pepmass')[0])
            charge2 = int(s2.get('params').get('charge').__str__()[0])
            precursor_feature2 = np.concatenate((self.gray_code(mass2), self.charge_to_one_hot(charge2)))
            precursor_feature_list2.append(precursor_feature2)

            label_list.append(l)

            tmp_precursor_feature_list1 = np.array(precursor_feature_list1) # n*34
            tmp_precursor_feature_list2 = np.array(precursor_feature_list2)

            intensList01 = np.array(peakslist1) # n*2449
            intensList02 = np.array(peakslist2)

            # Calculation of normalized dot product: reference_intensity:500*2449, ndp_r_spec_list: (500,), np.array(peakslist1):n*2449, np.array(ndp_spec_list1):(n,)
            tmp_dplist01 = self.caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1), np.array(ndp_spec_list1)) # n*500
            tmp_dplist02 = self.caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist2), np.array(ndp_spec_list2))

            label = np.array(label_list)
            label = np.array(label.reshape(label.shape[0], 1))

            tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
            tmp02 = concatenate((tmp_dplist02, intensList02), axis=1)
            spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)
            spectrum02 = concatenate((tmp02, tmp_precursor_feature_list2), axis=1)
            tmp_data = concatenate((spectrum01, spectrum02), axis=1)
            mgf_datasetTemp = concatenate((tmp_data, label), axis=1)
            mgf_datasetTempList.append(mgf_datasetTemp)

            # df = pd.DataFrame(self.mgf_dataset)
            # df.to_csv(saveName, mode="a+", header=False, index=False)

            #del self.mgf_dataset
            peakslist1.clear()
            peakslist2.clear()
            precursor_feature_list1.clear()
            precursor_feature_list2.clear()
            ndp_spec_list1.clear()
            ndp_spec_list2.clear()
            label_list.clear()
        self.mgf_dataset = np.vstack(mgf_datasetTempList)
        print("self.mgf_dataset.shape:",self.mgf_dataset.shape)
        end_time_transform_for = time.perf_counter()
        print("transform for loop run time：",end_time_transform_for - start_time_transform_for)

        del self.MGF
        print('Finish to calculate data set...')

    def __getitem__(self, item):
        return self.mgf_dataset[item]
    def __len__(self):
        return len(self.mgf_dataset)

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

    def get_bin_index(self, mz, min_mz, bin_size):
        relative_mz = mz - min_mz
        return max(0, int(np.floor(relative_mz / bin_size)))

    def bin_spectrum(self, mz_array, intensity_array, max_mz=2500, min_mz=50.5, bin_size=1.0005079):
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
            bin_index = self.get_bin_index(mz, min_mz, bin_size)

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
            logging.debug('zero intensity found')
        return results

    def caculate_nornalization_dp(self, reference, ndp_r_spec_list, bin_spectra, ndp_bin_sp):
        ndp_r_spec_list = ndp_r_spec_list.reshape(ndp_r_spec_list.shape[0], 1) # (500,1)
        ndp_bin_sp = ndp_bin_sp.reshape(ndp_bin_sp.shape[0], 1) # (n,1)
        tmp_dp_list = np.dot(bin_spectra, np.transpose(reference)) # (n,500)
        dvi = np.dot(ndp_bin_sp, np.transpose(ndp_r_spec_list)) # (n,500)
        result = tmp_dp_list / dvi  #(n,500)
        return result

class Dataset_RawDataset(data.dataset.Dataset):
    def __init__(self, data_file):
        if not os.path.exists(data_file):
            raise RuntimeError("Can not find mgf file: '%s'" % data_file)
        self.mgf_dataset = pd.read_csv(data_file, header=None, index_col=None).values
        print(self.mgf_dataset.shape)

    def __getitem__(self, item):
        return self.mgf_dataset[item]

    def __len__(self):
        return len(self.mgf_dataset)

class SiameseNetwork1(nn.Module):

    def __init__(self):
        super(SiameseNetwork1, self).__init__()

        self.fc1_1 = nn.Linear(34, 32)
        # self.fc1_1 = nn.Linear(61, 32)
        self.fc1_2 = nn.Linear(32, 5)

        self.cnn1 = nn.Conv1d(1, 30, 3)
        self.maxpool1 = nn.MaxPool1d(2)

        self.cnn2 = nn.Conv1d(1, 30, 3)
        self.maxpool2 = nn.MaxPool1d(2)

        self.fc2 = nn.Linear(1 * 44165, 32)
        # self.fc2 = nn.Linear(38165, 32)

        # self.dropout = nn.Dropout(0.01)

    def forward_once(self, preInfo, fragInfo, refSpecInfo):
        preInfo = self.fc1_1(preInfo)
        preInfo = F.selu(preInfo)
        preInfo = self.fc1_2(preInfo)
        preInfo = F.selu(preInfo)
        preInfo = preInfo.view(preInfo.size(0), -1)

        fragInfo = self.cnn1(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = self.maxpool1(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = fragInfo.view(fragInfo.size(0), -1)

        refSpecInfo = self.cnn2(refSpecInfo)
        refSpecInfo = F.selu(refSpecInfo)
        refSpecInfo = self.maxpool2(refSpecInfo)
        fragInfo = F.selu(fragInfo)
        refSpecInfo = refSpecInfo.view(refSpecInfo.size(0), -1)  # Change the shape of the data, -1 means indeterminate, it depends

        output = torch.cat((preInfo, fragInfo, refSpecInfo), 1)
        # output = self.dropout(output)
        output = self.fc2(output)
        output = F.selu(output)
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
        # input1_1 = spectrum01[:, :, :100]
        # input1_2 = spectrum01[:, :, 100:2549]
        # input1_3 = spectrum01[:, :, 2549:]
        #
        # input2_1 = spectrum02[:, :, :100]
        # input2_2 = spectrum02[:, :, 100:2549]
        # input2_3 = spectrum02[:, :, 2549:]

        refSpecInfo1, fragInfo1, preInfo1 = input1_3.cuda(), input1_2.cuda(), input1_1.cuda()
        refSpecInfo2, fragInfo2, preInfo2 = input2_3.cuda(), input2_2.cuda(), input2_1.cuda()

        output01 = self.forward_once(refSpecInfo1, fragInfo1, preInfo1)
        output02 = self.forward_once(refSpecInfo2, fragInfo2, preInfo2)

        return output01, output02

class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()

        self.fc1_1 = nn.Linear(34, 32)
        # self.fc1_1 = nn.Linear(34, 16)
        # self.fc1_1 = nn.Linear(61, 32)
        # self.fc1_2 = nn.Linear(16, 5)
        self.fc1_2 = nn.Linear(32, 5)

        self.cnn11 = nn.Conv1d(1, 30, 3)
        self.maxpool11 = nn.MaxPool1d(2)
        # self.cnn12 = nn.Conv1d(30, 30, 3)
        # self.maxpool12 = nn.MaxPool1d(6)
        # self.cnn13 = nn.Conv1d(30, 30, 3)
        # self.maxpool13 = nn.MaxPool1d(3)

        self.cnn21 = nn.Conv1d(1, 30, 3)
        self.maxpool21 = nn.MaxPool1d(2)
        self.cnn22 = nn.Conv1d(30, 30, 3)
        self.maxpool22 = nn.MaxPool1d(2)
        # self.cnn23 = nn.Conv1d(30, 30, 3)
        # self.maxpool23 = nn.MaxPool1d(6)
        # self.cnn24 = nn.Conv1d(30, 30, 3)
        # self.maxpool24 = nn.MaxPool1d(3)

        self.fc2 = nn.Linear(25775, 32)
        # self.fc2 = nn.Linear(19775, 32)

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
        refSpecInfo = refSpecInfo.view(refSpecInfo.size(0), -1)  # Change the shape of the data, -1 means indeterminate, it depends

        output = torch.cat((preInfo, fragInfo, refSpecInfo), 1)
        # output = self.dropout(output)
        output = self.fc2(output)
        # output = F.selu(output)
        return output

    def forward(self, spectrum01, spectrum02):

        spectrum01 = spectrum01.reshape(spectrum01.shape[0], 1, spectrum01.shape[1])
        spectrum02 = spectrum02.reshape(spectrum02.shape[0], 1, spectrum02.shape[1])

        # input1_1 = spectrum01[:, :, :50]
        # input1_2 = spectrum01[:, :, 50:2499]
        # input1_3 = spectrum01[:, :, 2499:]
        #
        # input2_1 = spectrum02[:, :, :50]
        # input2_2 = spectrum02[:, :, 50:2499]
        # input2_3 = spectrum02[:, :, 2499:]

        input1_1 = spectrum01[:, :, :500]
        input1_2 = spectrum01[:, :, 500:2949]
        input1_3 = spectrum01[:, :, 2949:]

        input2_1 = spectrum02[:, :, :500]
        input2_2 = spectrum02[:, :, 500:2949]
        input2_3 = spectrum02[:, :, 2949:]

        # input1_1 = spectrum01[:, :, :100]
        # input1_2 = spectrum01[:, :, 100:2549]
        # input1_3 = spectrum01[:, :, 2549:]
        #
        # input2_1 = spectrum02[:, :, :100]
        # input2_2 = spectrum02[:, :, 100:2549]
        # input2_3 = spectrum02[:, :, 2549:]

        refSpecInfo1, fragInfo1, preInfo1 = input1_3.cuda(), input1_2.cuda(), input1_1.cuda()
        refSpecInfo2, fragInfo2, preInfo2 = input2_3.cuda(), input2_2.cuda(), input2_1.cuda()

        output01 = self.forward_once(refSpecInfo1, fragInfo1, preInfo1)
        output02 = self.forward_once(refSpecInfo2, fragInfo2, preInfo2)

        return output01, output02

class SiameseNetwork3(nn.Module):

    def __init__(self):
        super(SiameseNetwork3, self).__init__()

        self.fc1_1 = nn.Linear(34,16)
        # self.fc1_1 = nn.Linear(34, 12)
        # self.fc1_2 = nn.Linear(32, 5)
        self.fc1_2 = nn.Linear(16, 5)

        # self.fc3 = nn.Linear(50, 32)
        self.fc3 = nn.Linear(100, 32)

        self.cnn21 = nn.Conv1d(1, 30, 3)
        self.maxpool21 = nn.MaxPool1d(2)
        self.cnn22 = nn.Conv1d(30, 30, 3)
        self.maxpool22 = nn.MaxPool1d(2)
        self.cnn23 = nn.Conv1d(30, 30, 3)
        self.maxpool23 = nn.MaxPool1d(2)

        self.dropout = nn.Dropout(0.001)

        # self.fc2 = nn.Linear(25775, 32)
        self.fc2 = nn.Linear(18337, 32)
        # self.fc2 = nn.Linear(36727, 32)
        # self.fc2 = nn.Linear(9157, 32)
        # self.fc2 = nn.Linear(19775, 32)

    def forward_once(self, preInfo, fragInfo, refSpecInfo):
        preInfo = self.fc1_1(preInfo)
        preInfo = F.selu(preInfo)
        preInfo = self.fc1_2(preInfo)
        preInfo = F.selu(preInfo)
        preInfo = preInfo.view(preInfo.size(0), -1)

        fragInfo = self.cnn21(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = self.maxpool21(fragInfo)
        # fragInfo = F.selu(fragInfo)
        fragInfo = self.cnn22(fragInfo)
        fragInfo = F.selu(fragInfo)
        fragInfo = self.maxpool22(fragInfo)
        # fragInfo = F.selu(fragInfo)
        # fragInfo = self.cnn23(fragInfo)
        # fragInfo = F.selu(fragInfo)
        # fragInfo = self.maxpool23(fragInfo)
        # fragInfo = F.selu(fragInfo)
        fragInfo = fragInfo.view(fragInfo.size(0), -1)

        refSpecInfo = self.fc3(refSpecInfo)
        refSpecInfo = F.selu(refSpecInfo)
        refSpecInfo = refSpecInfo.view(refSpecInfo.size(0), -1)  # Change the shape of the data, -1 means indeterminate, it depends

        output = torch.cat((preInfo, fragInfo, refSpecInfo), 1)
        # output = self.dropout(output)
        output = self.fc2(output)

        return output

    def forward(self, spectrum01, spectrum02):

        spectrum01 = spectrum01.reshape(spectrum01.shape[0], 1, spectrum01.shape[1])
        spectrum02 = spectrum02.reshape(spectrum02.shape[0], 1, spectrum02.shape[1])

        # input1_1 = spectrum01[:, :, :50]
        # input1_2 = spectrum01[:, :, 50:2499]
        # input1_3 = spectrum01[:, :, 2499:]
        #
        # input2_1 = spectrum02[:, :, :50]
        # input2_2 = spectrum02[:, :, 50:2499]
        # input2_3 = spectrum02[:, :, 2499:]

        input1_1 = spectrum01[:, :, :100]
        input1_2 = spectrum01[:, :, 100:2549]
        input1_3 = spectrum01[:, :, 2549:]

        input2_1 = spectrum02[:, :, :100]
        input2_2 = spectrum02[:, :, 100:2549]
        input2_3 = spectrum02[:, :, 2549:]

        refSpecInfo1, fragInfo1, preInfo1 = input1_3.cuda(), input1_2.cuda(), input1_1.cuda()
        refSpecInfo2, fragInfo2, preInfo2 = input2_3.cuda(), input2_2.cuda(), input2_1.cuda()

        output01 = self.forward_once(refSpecInfo1, fragInfo1, preInfo1)
        output02 = self.forward_once(refSpecInfo2, fragInfo2, preInfo2)

        return output01, output02

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        euclidean_distance = euclidean_distance.double()
        #print(euclidean_distance)
        label = label.double()
        loss_contrastive = torch.mean(label * euclidean_distance + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class PlotMyROC():
    def __init__(self):
        self.test = None

    def cal_all_rate(self, dis, label, thres):
        all_number = len(dis)
        # print all_number
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for item in range(all_number):
            disease = dis[item]
            if disease <= thres:
                disease = 1
            if disease == 1:
                if label[item] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if label[item] == 0:
                    TN += 1
                else:
                    FN += 1

        # print TP+FP+TN+FN
        accracy = float(TP + TN) / float(all_number)
        # print(accracy)
        if TP + FP == 0:
            precision = 0
        else:
            precision = float(TP) / float(TP + FP)

        TPR = float(TP) / float(TP + FN)
        TNR = float(TN) / float(FP + TN)
        FNR = float(FN) / float(TP + FN)
        FPR = float(FP) / float(FP + TN)
        # print accracy, precision, TPR, TNR, FNR, FPR
        return accracy, precision, TPR, TNR, FNR, FPR

    def my_roc(self, dis, label):
        # disease_class = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax']
        # style = ['r-', 'g-', 'b-', 'y-', 'r--', 'g--', 'b--', 'y--']
        style = ['r-']
        '''
        plot roc and calculate AUC/ERR, result: (prob, label) 
        '''
        # print(dis)
        threshold_vaule = sorted(dis)
        threshold_num = len(label)
        accracy_array = np.zeros(threshold_num)
        precision_array = np.zeros(threshold_num)
        TPR_array = np.zeros(threshold_num)
        TNR_array = np.zeros(threshold_num)
        FNR_array = np.zeros(threshold_num)
        FPR_array = np.zeros(threshold_num)
        for thres in range(len(label)):
            accracy, precision, TPR, TNR, FNR, FPR = self.cal_all_rate(dis, label, threshold_vaule[thres])
            accracy_array[thres] = accracy
            precision_array[thres] = precision
            TPR_array[thres] = TPR
            TNR_array[thres] = TNR
            FNR_array[thres] = FNR
            FPR_array[thres] = FPR

        AUC = np.trapz(TPR_array, FPR_array)
        threshold = np.argmin(abs(FNR_array - FPR_array))
        EER = (FNR_array[threshold] + FPR_array[threshold]) / 2
        ACC = accracy_array[threshold]
        print("Threshold:" + str(threshold_vaule[threshold]))

        return FPR_array, TPR_array, AUC, ACC, EER

def show_test02_roc(batch_size, fpr_list, tpr_list, auc_list, acc_list, eer_list, loss):
    # drawing
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    # FPR is the abscissa, TPR is the ordinate
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    # style = ['r-', 'g-', 'b-', 'y-', 'r--', 'g--', 'b--', 'y--']
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, alpha=0.7, label=u'Epoch%2d: AUC=%.3f; ACC=%.3f; EER=%0.3f;' % (i, auc_list[i], acc_list[i], eer_list[i]))
    plt.plot((0, 1), (0, 1), lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'ROC and AUC', fontsize=17)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(loss)), loss, 'r')
    plt.title('Loss for Iteration: %s Batch_size: %s' % (len(loss), batch_size))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    # plt.savefig('../Picture/' + saveName + '.jpg')
    plt.show()
    # plt.close()

    # plt.show()

def show_test01_roc(batch_size, fpr_list, tpr_list, auc_list, acc_list, eer_list, loss, saveName):
    # drawing
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(16, 8), dpi=1200)
    plt.subplot(1, 2, 1)
    # style = ['r-', 'g-', 'b-', 'y-', 'r--', 'g--', 'b--', 'y--']
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, alpha=0.7, label=u'Epoch%2d: AUC=%.3f; ACC=%.3f' % ((i+1)*10, auc_list[i], acc_list[i]))
    plt.plot((0, 1), (0, 1), lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=11)
    plt.title(u'(a) ROC and AUC', fontsize=13)
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(loss)), loss, 'r')
    plt.title('(b) Loss for Iteration: %s Batch_size: %s' % (len(loss), batch_size), fontsize=13)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('../Picture/' + saveName + '.jpg')
    # plt.show()
    plt.close()

    # plt.show()

if __name__ == '__main__':

    # Enter Epoches
    # epoches = input("Please input epoches:")
    epoches = 20
    epoches = int(epoches)

    # 输入Batch size
    # batch_size = input("Please input batch_size:")
    batch_size = 128
    batch_size = int(batch_size)

    # The name of the saved training image
    # figName = input("Please picture's name:")
    figName = "GNM500R200301-080802"

    prepareData_time = time.perf_counter()

    # Basic training parameters
    train_batch_size = batch_size
    
    test_batch_size = 1

    # train_dataset = MGFDataSet('../Data/PRIDE_Exp_Complete_Ac_1653.pride.mgf', '../Data/train.csv', 500)
    # test_dataset = MGFDataSet('../Data/PRIDE_Exp_Complete_Ac_1653.pride.mgf', '../Data/test.csv', 500)

    torch.multiprocessing.set_sharing_strategy('file_system')

    t1 = time.perf_counter()

    # RawDataSet('../SpectraPairsData/0623_all_spectra_mgf_bins.mgf', '../SpectraPairsData/0623_NP_Sample_SP_Shuf.csv', "0628_NP_HMASS_100RF_Coded.txt")

    # RawDataSet('../SpectraPairsData/0722_all_bins.mgf', '../SpectraPairsData/0722_NP_Sample_Result_Shuf.csv', "0807_NP_NM_2449Bins_100RF_Coded.txt")
    # RawDataSet('../SpectraPairsData/0722_all_bins.mgf', '../SpectraPairsData/0722_NP_Sample_Result_Shuf.csv', "0805_NP_NM_2449Bins_50RF_Coded.txt")
    # RawDataSet('../SpectraPairsData/0722_all_bins.mgf', '../SpectraPairsData/0722_NP_Sample_Result_Shuf.csv', "0807_NP_NM_2449Bins_500RF_Coded.txt")

    # ！！！ In Github, we do not provide the MgfBin_YEAST.mgf file because it is too large and exceeds the push limit
    train_dataset = RawDataSet('./Data/MgfBin_YEAST.mgf', './Data/NP_SpectraPairsBin_YEAST_TRAIN_200K.csv', "./0807_NP_NM_2449Bins_100RF_Coded_Train.txt")
    test_dataset = RawDataSet('./Data/MgfBin_YEAST.mgf', './Data/NP_SpectraPairsBin_YEAST_TEST_200K.csv', "./0807_NP_NM_2449Bins_100RF_Coded_Test.txt")
    
    # train_dataset = Dataset_RawDataset("../Data/test-train.txt")
    # test_dataset = Dataset_RawDataset("../Data/test-test.txt")
    #
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)

    t2 = time.perf_counter()

    print("Load Data Done! Use Time: {}".format(t2 - t1))

    # net = SiameseNetwork3().double()
    # net = SiameseNetwork1().double()
    net = SiameseNetwork2().double()
    net = net.cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)

    counter, loss_history, acc = [], [], []
    fpr_list, tpr_list, auc_list, acc_list, eer_list = [], [], [], [], []
    iteration_number = 0

    start_time = time.perf_counter()
    plt_roc = PlotMyROC()

    for epoch in range(0, epoches):
        for i, data in enumerate(train_dataloader, 0):
            # spec0, spec1 = data[:, :3010], data[:, 3010:-1]
            spec0, spec1 = data[:, :2983], data[:, 2983:-1]
            # spec0, spec1 = data[:, :2533], data[:, 2533:-1]
            # spec0, spec1 = data[:, :2583], data[:, 2583:-1]
            # spec0, spec1 = data[:, :2610], data[:, 2610:-1]
            label = data[:, -1]
            optimizer.zero_grad()
            output1, output2 = net(spec0, spec1)
            label = label.cuda()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 100 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
        tmp_epoch = epoch + 1
        if tmp_epoch % 10 == 0:
            with torch.no_grad():
                dis_list, flag_list = [], []
                for j, test_data in enumerate(test_dataloader, 0):
                    # test_spec0, test_spec1 = test_data[:, :3010], test_data[:, 3010:-1]
                    test_spec0, test_spec1 = test_data[:, :2983], test_data[:, 2983:-1]
                    # test_spec0, test_spec1 = test_data[:, :2533], test_data[:, 2533:-1]
                    # test_spec0, test_spec1 = test_data[:, :2583], test_data[:, 2583:-1]
                    # test_spec0, test_spec1 = test_data[:, :2610], test_data[:, 2610:-1]
                    test_label = test_data[:, -1]
                    out1, out2 = net(test_spec0, test_spec1)
                    euclidean_distance = F.pairwise_distance(out1, out2)
                    euclidean_dis = float(euclidean_distance.cpu().data.numpy()[0])
                    dis_list.append(euclidean_dis**2)
                    flag_list.append(float(test_label.numpy()[0]))
                net.train()
                fpr, tpr, auc, acc, eer = plt_roc.my_roc(dis_list, flag_list)
                dis_list.clear()
                flag_list.clear()
                fpr_list.append(fpr)
                tpr_list.append(tpr)
                auc_list.append(auc)
                acc_list.append(acc)
                eer_list.append(eer)
                print("fpr_listL:",fpr_list.tolist())
                print("tpr_list:",tpr_list.tolist())
                print("auc_list:",auc_list.tolist())
                print("acc_list:",acc_list.tolist())
                print("eer_list:",eer_list.tolist())
                print("loss_history:",loss_history)
                del fpr, tpr, auc, acc, eer

    # show_test01_roc(train_batch_size, fpr_list, tpr_list, auc_list, acc_list, eer_list, loss_history, figName)


    # torch.save(net, './model/081501_15_1000_NM100R_model.pkl')

    # show_plot(train_batch_size, acc, loss_history, figName)

    end_time = time.perf_counter()
    print("Use Time: {}".format(end_time - start_time))
    print("Total Time:",(end_time-t1))

