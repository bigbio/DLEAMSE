#! -*- codinf:utf-8 -*-
import time

import pandas as pd
import numpy as np
import torch
from numba import njit
from pyteomics.mgf import read
from pyteomics.mgf import read_header

"""
This script is used to compare the use-time of NDP and DLEAMS
"""


@njit
def caculate_spec(bin_spec):
  ndp_spec = np.math.sqrt(np.dot(bin_spec, bin_spec))
  return ndp_spec


@njit
def get_bin_index(mz, min_mz, bin_size):
  relative_mz = mz - min_mz
  return max(0, int(np.floor(relative_mz / bin_size)))


def ndp_bin_spectrum(mz_array, intensity_array):
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
  max_mz = int(2500)
  min_mz = float(50.5)
  bin_size = float(1.0005079)
  # max_mz = int(1995)
  # min_mz = float(84)
  # bin_size = float(1)

  nbins = int(float(max_mz - min_mz) / float(bin_size)) + 1

  results_dict = {}
  results = np.zeros(nbins)
  final_results = np.zeros(nbins)

  for index in range(len(mz_array)):
    mz = float(mz_array[index])
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

  # 取出前100个最高度的峰
  # print(results)
  # print(results)

  # for i in range(results.shape[0]):
  #     results_dict[i] = results[i]
  #     final_results[i] = 0

  results_tensor = torch.from_numpy(results)
  results_tensor = results_tensor.cuda()
  test_topk = torch.topk(results_tensor, k=100)
  top100_intens = np.array(test_topk[0].cpu())
  top100_index = np.array(test_topk[1].cpu())

  for i in range(top100_index.shape[0]):
    final_results[top100_index[i]] = top100_intens[i]

  return final_results


def caculate_nornalization_dp(bin_spectrum01, bin_spectrum02):
  tmp_01 = caculate_spec(bin_spectrum01)
  tmp_02 = caculate_spec(bin_spectrum02)
  dvi = np.dot(tmp_01, tmp_02)
  tmp_dp_list = np.dot(bin_spectrum01, bin_spectrum02)
  result = tmp_dp_list / dvi
  return result


def calculate_ndp_time(spectra_mgf_file1, spectra_mgf_file2):
  score_list = []

  bins_spectrum_01, bins_spectrum_02 = [], []

  tmp_time_01 = time.perf_counter()

  spectra01 = read(spectra_mgf_file1, convert_arrays=1)
  spectra02 = read(spectra_mgf_file2, convert_arrays=1)

  for data01 in spectra01:
    spectrum01_mz_array = data01.get("m/z array")
    spectrum01_intens_array = data01.get("intensity array")
    bin_spectrum01 = ndp_bin_spectrum(spectrum01_mz_array, spectrum01_intens_array)
    bins_spectrum_01.append(bin_spectrum01)

  for data02 in spectra02:
    spectrum02_mz_array = data02.get("m/z array")
    spectrum02_intens_array = data02.get("intensity array")
    bin_spectrum02 = ndp_bin_spectrum(spectrum02_mz_array, spectrum02_intens_array)
    bins_spectrum_02.append(bin_spectrum02)

  time01 = time.perf_counter()
  print("两文件编码所用的时间为：{}".format(time01 - tmp_time_01))

  for j in range(len(bins_spectrum_01)):
    score = caculate_nornalization_dp(bins_spectrum_01[j], bins_spectrum_02[j])
    score_list.append(score)
  # np.savetxt("./data/1130_test_use_time_ndp.txt", score_list)
  time02 = time.perf_counter()
  print("Similarity use time: {}".format(time02 - time01))


if __name__ == '__main__':
  print("test")

  time_01 = time.perf_counter()
  # 首先是定义代码的输入，需要输入谱图对数据，然后需要数据谱图对数据对应的mgf文件
  # spectra_pairs_file = "./data/062401_test_ups_specs_BC_NFTR_NFTR_NF_None_TR_None_PPR_None_CHR_givenCharge_PRECTOL_3.0_binScores.txt"

  # spectra_mgf_file1 = "./data/0622_Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc_SCAN.mgf"
  # spectra_mgf_file2 = "./data/0622_Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc_SCAN.mgf"

  # spectra_mgf_file1 = "./data/OEI04195.mgf"
  # spectra_mgf_file2 = "./data/OEI04195.mgf"
  # spectra_mgf_file1 = "./data/test50000.mgf"
  # spectra_mgf_file2 = "./data/test50000.mgf"
  # spectra_mgf_file1 = "./data/crap.mgf"
  # spectra_mgf_file2 = "./data/crap.mgf"

  # spectra_mgf_file1 = "./data/sample10000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample10000_mgf.mgf"

  # spectra_mgf_file1 = "./data/sample20000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample20000_mgf.mgf"
  #
  # spectra_mgf_file1 = "./data/sample40000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample40000_mgf.mgf"
  #
  # spectra_mgf_file1 = "./data/sample80000_mgf.mgf"
  # spectra_mgf_file2 = "./data/sample80000_mgf.mgf"
  spectra_mgf_file1 = "./data/crap_40000_mgf.mgf"
  spectra_mgf_file2 = "./data/crap_40000_mgf.mgf"

  # spectra_mgf_file1 = "../SimilarityScoring/data/before_0622/Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc.mgf"
  # spectra_mgf_file2 = "../SimilarityScoring/data/before_0622/Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc.mgf"

  tmp_time_00 = time.perf_counter()
  calculate_ndp_time(spectra_mgf_file1, spectra_mgf_file2)
  time_02 = time.perf_counter()
  print("不计算文件加载，仅计算编码和计算NDP的总时间：{}".format(time_02 - tmp_time_00))
  print("Total use time: {}".format(time_02 - time_01))
