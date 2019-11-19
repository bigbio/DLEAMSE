#! -*- codinf:utf-8 -*-
"""
created by qincy
"""

import time
import pandas as pd
import numpy as np
from numba import njit
from pyteomics.mgf import read


"""
This script is used to compare the use-time of Pearson and DSMapper
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

    #取出前100个最高度的峰
    # print(results)
    # print(results)
    for i in range(results.shape[0]):
        results_dict[i] = results[i]
        final_results[i] = 0

    tmp_result = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    result = tmp_result[0:99]
    for rst in result:
        final_results[rst[0]] = rst[1]

    return final_results
def caculate_nornalization_dp(bin_spectrum01, bin_spectrum02):

    tmp_01 = caculate_spec(bin_spectrum01)
    tmp_02 = caculate_spec(bin_spectrum02)
    dvi = np.dot(tmp_01, tmp_02)
    tmp_dp_list = np.dot(bin_spectrum01, bin_spectrum02)
    result = tmp_dp_list / dvi
    return result

def calculate_ndp_time(spectra01, spectra02, mgf_01: dict, mgf_02: dict):

    score_list = []
    # print(mgf_01.get(spectra01[1]))
    # print(type(mgf_01.get(spectra01[1])))
    bins_spectrum_01, bins_spectrum_02 = [], []

    for i in range(spectra01.shape[0]):
        spectrum01_mz_array = mgf_01.get(spectra01[i]).get("m/z array")
        spectrum01_intens_array = mgf_01.get(spectra01[i]).get("intensity array")
        spectrum02_mz_array = mgf_02.get(spectra02[i]).get("m/z array")
        spectrum02_intens_array = mgf_02.get(spectra02[i]).get("intensity array")
        bin_spectrum01 = ndp_bin_spectrum(spectrum01_mz_array, spectrum01_intens_array)
        bin_spectrum02 = ndp_bin_spectrum(spectrum02_mz_array, spectrum02_intens_array)
        bins_spectrum_01.append(bin_spectrum01)
        bins_spectrum_02.append(bin_spectrum02)

    time01 = time.perf_counter()
    for j in range(len(bins_spectrum_01)):
        score = caculate_nornalization_dp(bins_spectrum_01[j], bins_spectrum_02[j])
        score_list.append(score)
    np.savetxt("./data/091801_test_use_time_ndp.txt", score_list)
    time02 = time.perf_counter()
    print("Use time: {}".format(time02 - time01))

if __name__ == '__main__':

    time_01 = time.perf_counter()
    #首先是定义代码的输入，需要输入谱图对数据，然后需要数据谱图对数据对应的mgf文件
    spectra_pairs_file = "./data/062401_test_ups_specs_BC_NFTR_NFTR_NF_None_TR_None_PPR_None_CHR_givenCharge_PRECTOL_3.0_binScores.txt"
    spectra_mgf_file1 = "./data/0622_Orbi2_study6a_W080314_6E008_yeast_S48_ft8_pc_SCAN.mgf"
    spectra_mgf_file2 = "./data/0622_Orbi2_study6a_W080314_6QC1_sigma48_ft8_pc_SCAN.mgf"

    spectra_pairs_data = pd.read_csv(spectra_pairs_file, sep="\t", header=None, index_col=None)

    spectra01 = spectra_pairs_data[0]
    spectra02 = spectra_pairs_data[3]

    spectra_mgf_data1 = read(spectra_mgf_file1)
    spectra_mgf_data2 = read(spectra_mgf_file2)
    mgf_01, mgf_02 = {}, {}
    for mgf01 in spectra_mgf_data1:
        mgf_01[mgf01.get('params').get('title')] = mgf01
    for mgf02 in spectra_mgf_data2:
        mgf_02[mgf02.get('params').get('title')] = mgf02

    calculate_ndp_time(spectra01, spectra02, mgf_01, mgf_02)
    time_02 = time.perf_counter()
    print("Total use time: {}".format(time_02 - time_01))
