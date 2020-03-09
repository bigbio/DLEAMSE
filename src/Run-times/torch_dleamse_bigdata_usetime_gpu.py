# -*- coding:utf-8 -*-
"""
This is a search program!
Create by qincy, April 17,2019
"""

import os
import logging
import time

import more_itertools
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
    refSpecInfo = refSpecInfo.view(refSpecInfo.size(0), -1)  # 改变数据的形状，-1表示不确定，视情况而定

    output = torch.cat((preInfo, fragInfo, refSpecInfo), 1)
    # output = self.dropout(output)
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


class RawDataSet01:

  def __init__(self, spectra_pairs_num):

    self.len = spectra_pairs_num
    self.mgf_dataset = None

  def transform(self, spectrum_list, reference_intensity):

    # 先将500个参考谱图的点积结果计算出来
    ndp_r_spec_list = caculate_r_spec(reference_intensity)

    peakslist1, precursor_feature_list1 = [], []
    ndp_spec_list = []
    encode_batch = 10000
    i, j, k = 0, 0, 0
    self.MGF = read(spectrum_list, convert_arrays=1)
    if encode_batch > self.len:
      for s1 in self.MGF:
        bin_s1 = bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
        # ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
        ndp_spec1 = caculate_spec(bin_s1)
        peakslist1.append(bin_s1)
        ndp_spec_list.append(ndp_spec1)
        mass1 = float(s1.get('params').get('pepmass')[0])
        charge1 = int(s1.get('params').get('charge').__str__()[0])
        precursor_feature1 = np.concatenate((self.gray_code(mass1), self.charge_to_one_hot(charge1)))
        precursor_feature_list1.append(precursor_feature1)

      tmp_precursor_feature_list1 = np.array(precursor_feature_list1)
      intensList01 = np.array(peakslist1)

      # 归一化点积的计算
      tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1),
                                               np.array(ndp_spec_list))
      tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
      spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

      self.mgf_dataset = spectrum01
      peakslist1.clear()
      precursor_feature_list1.clear()
      ndp_spec_list.clear()
    else:
      for s1 in self.MGF:
        bin_s1 = bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
        # ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
        ndp_spec1 = caculate_spec(bin_s1)
        peakslist1.append(bin_s1)
        ndp_spec_list.append(ndp_spec1)
        mass1 = float(s1.get('params').get('pepmass')[0])
        charge1 = int(s1.get('params').get('charge').__str__()[0])
        precursor_feature1 = np.concatenate((self.gray_code(mass1), self.charge_to_one_hot(charge1)))
        precursor_feature_list1.append(precursor_feature1)

        if len(peakslist1) == encode_batch:
          i += 1
          tmp_precursor_feature_list1 = np.array(precursor_feature_list1)
          intensList01 = np.array(peakslist1)

          # 归一化点积的计算
          tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1),
                                                   np.array(ndp_spec_list))

          tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
          spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

          if i == 1:
            self.mgf_dataset = spectrum01
          else:
            tt = time.perf_counter()
            self.mgf_dataset = np.vstack((self.mgf_dataset, spectrum01))
            tt_end = time.perf_counter()
            print(tt_end - tt)
          peakslist1.clear()
          precursor_feature_list1.clear()
          ndp_spec_list.clear()
          j = i * encode_batch

        elif (j + encode_batch) > self.len:
          if len(peakslist1) == self.len - j:
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

    return self.mgf_dataset

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


class Dataset_RawDataset(data.dataset.Dataset):
  def __init__(self, data):
    self.mgf_dataset = data

  def __getitem__(self, item):
    return self.mgf_dataset[item]

  def __len__(self):
    return self.mgf_dataset.shape[0]


@njit
def caculate_spec(bin_spec):
  ndp_spec1 = np.math.sqrt(np.dot(bin_spec, bin_spec))
  return ndp_spec1


@njit
def caculate_r_spec(reference_intensity):
  ndp_r_spec_list = np.zeros(500)
  # ndp_r_spec_list = np.zeros(100)
  for x in range(500):
    # for x in range(100):
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
  ndp_r_spec_list = ndp_r_spec_list.reshape(ndp_r_spec_list.shape[0], 1)
  ndp_bin_sp = ndp_bin_sp.reshape(ndp_bin_sp.shape[0], 1)
  tmp_dp_list = np.dot(bin_spectra, np.transpose(reference))
  dvi = np.dot(ndp_bin_sp, np.transpose(ndp_r_spec_list))
  result = tmp_dp_list / dvi
  return result


def embedding_dataset(net, spectrum_list, reference_intensity, spectra_pairs_num):
  out_list = None

  # net = torch.load(model)
  # 生成RAW文件，在存在的情况下不需要重复执行
  tmp_time_01 = time.perf_counter()
  rdataset01 = RawDataSet01(spectra_pairs_num)
  vstack_data = rdataset01.transform(spectrum_list, reference_intensity)
  tmp_time_02 = time.perf_counter()
  print("enconding use time : {}".format(tmp_time_02 - tmp_time_01))

  dataset = Dataset_RawDataset(vstack_data)
  print(dataset.__len__())
  dataloader = data.DataLoader(dataset=dataset, batch_size=192, shuffle=False, num_workers=1)
  # tmp_time_03 = time.perf_counter()
  # print("load file use time : {}".format(tmp_time_03 - tmp_time_02))

  for j, test_data in enumerate(dataloader, 0):
    # todo:放spectrum_title在文件的前方

    spectrum01 = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])

    input1_1 = spectrum01[:, :, :500]
    input1_2 = spectrum01[:, :, 500:2949]
    input1_3 = spectrum01[:, :, 2949:]

    refSpecInfo1, fragInfo1, preInfo1 = input1_3.cuda(), input1_2.cuda(), input1_1.cuda()
    output01 = net.forward_once(refSpecInfo1, fragInfo1, preInfo1)
    # output01 = net.forward_once(input1_3, input1_2, input1_1)
    # out1 = output01.detach().numpy()
    out1 = output01.cpu().detach().numpy()
    if j == 0:
      out_list = out1
    else:
      out_list = np.vstack((out_list, out1))

  tmp_time_03_1 = time.perf_counter()
  print("embeding use time : {}".format(tmp_time_03_1 - tmp_time_02))
  return out_list


def calculate_dsmapper_time(spectra_mgf_file1, spectra_mgf_file2):
  score_list = []
  # model = "../SpectraPairsData/080802_20_1000_NM500R_model.pkl"
  model = "./data/080802_20_1000_NM500R_model.pkl"

  tmp_time_01 = time.perf_counter()
  net = torch.load(model)
  tmp_time_02 = time.perf_counter()
  print("加载模型用时：{}".format(tmp_time_02 - tmp_time_01))

  # 五百个参考的谱图
  # reference_spectra = read("./0715_50_rf_spectra.mgf", convert_arrays=1)
  reference_spectra = read("../SpectraPairsData/0722_500_rf_spectra.mgf", convert_arrays=1)
  # reference_spectra = read("./data/0722_500_rf_spectra.mgf", convert_arrays=1)
  reference_intensity = np.array(
    [bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in reference_spectra])

  spectra_pairs_num = more_itertools.ilen(read(spectra_mgf_file1, convert_arrays=1))
  tmp_time_03 = time.perf_counter()
  print("准备相关数据用时：{}".format(tmp_time_03 - tmp_time_02))

  embedded_01 = embedding_dataset(net, spectra_mgf_file1, reference_intensity, spectra_pairs_num)
  embedded_02 = embedding_dataset(net, spectra_mgf_file2, reference_intensity, spectra_pairs_num)

  # embedded_01 = embedded_01.reshape(embedded_01.shape[0], 1, embedded_01.shape[1])
  # embedded_02 = embedded_02.reshape(embedded_02.shape[0], 1, embedded_02.shape[1])

  time01 = time.perf_counter()
  print("数据编码加嵌入的总用时：{}".format(time01 - tmp_time_03))

  for i in range(embedded_01.shape[0]):
    score = np.linalg.norm(embedded_01[i] - embedded_02[i])
    score_list.append(score)
  # np.savetxt("./data/091801_test_use_time_dsmapper.txt", score_list)
  time02 = time.perf_counter()
  print("calc_EU use time: {}".format(time02 - time01))


if __name__ == '__main__':
  print("test")
  time_01 = time.perf_counter()
  # 首先是定义代码的输入，需要输入谱图对数据，然后需要数据谱图对数据对应的mgf文件
  # spectra_pairs_file = "./data/062401_test_ups_specs_BC_NFTR_NFTR_NF_None_TR_None_PPR_None_CHR_givenCharge_PRECTOL_3.0_binScores.txt"
  # spectra_mgf_file1 = "./data/0622_Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc_SCAN.mgf"
  # spectra_mgf_file2 = "./data/0622_Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc_SCAN.mgf"
  # spectra_mgf_file1 = "./data/test50000.mgf"
  # spectra_mgf_file2 = "./data/test50000.mgf"
  # spectra_mgf_file1 = "./data/OEI04195.mgf"
  # spectra_mgf_file2 = "./data/OEI04195.mgf"
  # spectra_mgf_file1 = "./data/crap.mgf"
  # spectra_mgf_file2 = "./data/crap.mgf"
  spectra_mgf_file1 = "./data/crap_40000_mgf.mgf"
  spectra_mgf_file2 = "./data/crap_40000_mgf.mgf"
  # spectra_mgf_file1 = "./data/test100000.mgf"
  # spectra_mgf_file2 = "./data/test100000.mgf"

  # spectra_mgf_file1 = "./data/sample10000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample10000_mgf.mgf"

  # spectra_mgf_file1 = "./data/sample20000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample20000_mgf.mgf"

  # spectra_mgf_file1 = "./data/sample40000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample40000_mgf.mgf"
  #
  # spectra_mgf_file1 = "./data/sample80000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample80000_mgf.mgf"

  # # 五百个参考的谱图
  # # reference_spectra = read("./0715_50_rf_spectra.mgf", convert_arrays=1)
  # reference_spectra = read("../SpectraPairsData/0722_500_rf_spectra.mgf", convert_arrays=1)
  # # reference_spectra = read("./data/0722_500_rf_spectra.mgf", convert_arrays=1)
  # reference_intensity = np.array([bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in reference_spectra])
  #
  # spectra_pairs_num = more_itertools.ilen(read(spectra_mgf_file1, convert_arrays=1))

  # spectra_mgf_file1 = "../SimilarityScoring/data/before_0622/Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc.mgf"
  # spectra_mgf_file2 = "../SimilarityScoring/data/before_0622/Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc.mgf"
  # spectra_mgf_file2 = "./data/0622_Orbi2_study6a_W080314_6QC1_sigma48_ft8_pc_SCAN.mgf"

  # spectra_pairs_data = pd.read_csv(spectra_pairs_file, sep="\t", header=None, index_col=None)
  #
  # spectra01 = spectra_pairs_data[0]
  # spectra02 = spectra_pairs_data[3]
  #
  # spectra_mgf_data1 = read(spectra_mgf_file1)
  # spectra_mgf_data2 = read(spectra_mgf_file2)
  # mgf_01, mgf_02 = {}, {}
  # for mgf01 in spectra_mgf_data1:
  #     mgf_01[mgf01.get('params').get('title')] = mgf01
  # for mgf02 in spectra_mgf_data2:
  #     mgf_02[mgf02.get('params').get('title')] = mgf02

  tmp_time_00 = time.perf_counter()
  # calculate_dsmapper_time(spectra_mgf_file1, spectra_mgf_file1, reference_intensity, spectra_pairs_num)
  calculate_dsmapper_time(spectra_mgf_file1, spectra_mgf_file1)
  time_02 = time.perf_counter()
  print("编码和嵌入和计算相似性的总用时：{}".format(time_02 - tmp_time_00))
  print("Total use time: {}".format(time_02 - time_01))
