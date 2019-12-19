# -*- coding:utf-8 -*-
"""
Encode spectra.
"""

import argparse
import os

from pyteomics.mgf import read

import pandas as pd
import numpy as np
from numpy import concatenate
from numba import njit

class EncodeDataset():

    def __init__(self, mgf_file, ref_spectra, miss_saveName):
        if not os.path.exists(mgf_file):
            raise RuntimeError("Can not find mgf file: '%s'" % mgf_file)

        self.len = len(read(mgf_file, convert_arrays=1).__iter__())
        self.data = self.transform(mgf_file, ref_spectra, miss_saveName)

    def transform(self, mgf_file, ref_spectra, miss_saveName):
        self.mgf_dataset = None
        print('Start to calculate data set...')
        #五百个参考的谱图
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
                    tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1), np.array(ndp_spec_list))

                    tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
                    spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

                    self.mgf_dataset = np.vstack((self.mgf_dataset, spectrum01))

                    peakslist1.clear()
                    precursor_feature_list1.clear()
                    ndp_spec_list.clear()
                else:
                    continue

        np_mr = np.array(charge_none_list)
        df_mr = pd.DataFrame(np_mr, index=None, columns=None)
        df_mr.to_csv(miss_saveName)

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

def encode_spectra(input, refrence_spectra,miss_record, output):
    """
    :param input: get .mgf file as input
    :param refrence_spectra: get a .mgf file contained 500 spectra as referece spectra from normalized dot product calculation
    :param miss_record: record title of some spectra which loss charge attribute
    :param output: a file for save the final encode information
    :return: None
    """
    vstack_data = EncodeDataset(input, refrence_spectra,miss_record).getData()
    np.savetxt(output, vstack_data)

