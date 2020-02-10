# -*- coding:utf-8 -*-
# @Author : Chunyuan Qin
"""
Encode spectra.
"""

import argparse
import os

import more_itertools
from pyteomics.mgf import read as mgf_read
from pyteomics.mzml import read as mzml_read

import pandas as pd
import numpy as np
from numpy import concatenate
from numba import njit

class EncodeDataset():

    def __init__(self, input_specta_num):
        self.len = input_specta_num
        self.spectra_dataset = None
        
    def transform_mgf(self, input_spctra_file, ref_spectra, miss_saveName):
        self.spectra_dataset = None
        print('Start spectra encoding ...')
        #五百个参考的谱图
        reference_spectra = mgf_read(ref_spectra, convert_arrays=1)
        reference_intensity = np.array([bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in reference_spectra])

        # 先将500个参考谱图的点积结果计算出来
        ndp_r_spec_list = caculate_r_spec(reference_intensity)

        peakslist1, precursor_feature_list1 = [], []
        ndp_spec_list = []
        i, j, k = 0, 0, 0
        charge_none_record, charge_none_list = 0, []
        encode_batch = 10000
        
        self.MGF = mgf_read(input_spctra_file, convert_arrays=1)
        if encode_batch > self.len:
            for s1 in self.MGF:
                
                # missing charge
                if s1.get('params').get('charge').__str__()[0] == "N":
                    charge_none_record += 1
                    spectrum_id = s1.get('params').get('title')
                    charge_none_list.append(spectrum_id)
                    continue
                else:
                    charge1 = int(s1.get('params').get('charge').__str__()[0])
                
                bin_s1 = bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
                # ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
                ndp_spec1 = caculate_spec(bin_s1)
                peakslist1.append(bin_s1)
                ndp_spec_list.append(ndp_spec1)
                mass1 = float(s1.get('params').get('pepmass')[0])
                # charge1 = int(s1.get('params').get('charge').__str__()[0])
                precursor_feature1 = np.concatenate((self.gray_code(mass1), self.charge_to_one_hot(charge1)))
                precursor_feature_list1.append(precursor_feature1)

            tmp_precursor_feature_list1 = np.array(precursor_feature_list1)
            intensList01 = np.array(peakslist1)

            # 归一化点积的计算
            tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1),
                                                     np.array(ndp_spec_list))
            tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
            spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

            self.spectra_dataset = spectrum01
            peakslist1.clear()
            precursor_feature_list1.clear()
            ndp_spec_list.clear()
        else:
            for s1 in self.MGF:

                # missing charge
                if s1.get('params').get('charge').__str__()[0] == "N":
                    charge_none_record += 1
                    spectrum_id = s1.get('params').get('title')
                    charge_none_list.append(spectrum_id)
                    continue
                else:
                    charge1 = int(s1.get('params').get('charge').__str__()[0])
                    
                bin_s1 = bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
                # ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
                ndp_spec1 = caculate_spec(bin_s1)
                peakslist1.append(bin_s1)
                ndp_spec_list.append(ndp_spec1)
                mass1 = float(s1.get('params').get('pepmass')[0])
                # charge1 = int(s1.get('params').get('charge').__str__()[0])
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
                        self.spectra_dataset = spectrum01
                    else:
                        self.spectra_dataset = np.vstack((self.spectra_dataset, spectrum01))
                    peakslist1.clear()
                    precursor_feature_list1.clear()
                    ndp_spec_list.clear()
                    j = i * encode_batch

                elif (j + encode_batch + charge_none_record) > self.len:
                    if len(peakslist1) == self.len - j - charge_none_record:
                        tmp_precursor_feature_list1 = np.array(precursor_feature_list1)
                        intensList01 = np.array(peakslist1)

                        # 归一化点积的计算
                        tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list,
                                                                 np.array(peakslist1), np.array(ndp_spec_list))

                        tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
                        spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

                        self.spectra_dataset = np.vstack((self.spectra_dataset, spectrum01))

                        peakslist1.clear()
                        precursor_feature_list1.clear()
                        ndp_spec_list.clear()
                    else:
                        continue

        if len(charge_none_list) > 0:
            np_mr = np.array(charge_none_list)
            df_mr = pd.DataFrame(np_mr, index=None, columns=None)
            df_mr.to_csv(miss_saveName)
            print("Charge Missing Number:{}".format(charge_none_record))
            del charge_none_list


        return self.spectra_dataset

    def transform_mzml(self, input_spctra_file, ref_spectra, miss_saveName):
        self.spectra_dataset = None
        print('Start spectra encoding ...')
        #五百个参考的谱图
        reference_spectra = mgf_read(ref_spectra, convert_arrays=1)
        reference_intensity = np.array([bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in reference_spectra])

        # 先将500个参考谱图的点积结果计算出来
        ndp_r_spec_list = caculate_r_spec(reference_intensity)

        peakslist1, precursor_feature_list1 = [], []
        ndp_spec_list = []
        i, j, k = 0, 0, 0
        charge_none_record, charge_none_list = 0, []
        encode_batch = 10000

        self.MZML = mzml_read(input_spctra_file)
        if encode_batch > self.len:
            for s1 in self.MZML:

                # missing charge
                if s1.get("precursorList").get("precursor")[0].get("selectedIonList").get("selectedIon")[0].get("charge state").__str__()[0] == "N":
                    charge_none_record += 1
                    spectrum_id = s1.get("spectrum title")
                    charge_none_list.append(spectrum_id)
                    continue
                else:
                    charge1 = int(s1.get("precursorList").get("precursor")[0].get("selectedIonList").get("selectedIon")[0].get("charge state").__str__()[0])

                bin_s1 = bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
                # ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
                ndp_spec1 = caculate_spec(bin_s1)
                peakslist1.append(bin_s1)
                ndp_spec_list.append(ndp_spec1)
                mass1 = s1.get("precursorList").get("precursor")[0].get("selectedIonList").get("selectedIon")[0].get(
                    "selected ion m/z")
                # mass1 = float(s1.get('params').get('pepmass')[0])
                # charge1 = int(s1.get('params').get('charge').__str__()[0])
                precursor_feature1 = np.concatenate((self.gray_code(mass1), self.charge_to_one_hot(charge1)))
                precursor_feature_list1.append(precursor_feature1)

            tmp_precursor_feature_list1 = np.array(precursor_feature_list1)
            intensList01 = np.array(peakslist1)

            # 归一化点积的计算
            tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1),
                                                     np.array(ndp_spec_list))
            tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
            spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

            self.spectra_dataset = spectrum01
            peakslist1.clear()
            precursor_feature_list1.clear()
            ndp_spec_list.clear()
        else:
            for s1 in self.MZML:

                # missing charge
                if s1.get("precursorList").get("precursor")[0].get("selectedIonList").get("selectedIon")[0].get(
                        "charge state").__str__()[0] == "N":
                    charge_none_record += 1
                    spectrum_id = s1.get("spectrum title")
                    charge_none_list.append(spectrum_id)
                    continue
                else:
                    charge1 = int(
                        s1.get("precursorList").get("precursor")[0].get("selectedIonList").get("selectedIon")[0].get(
                            "charge state").__str__()[0])

                bin_s1 = bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
                # ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
                ndp_spec1 = caculate_spec(bin_s1)
                peakslist1.append(bin_s1)
                ndp_spec_list.append(ndp_spec1)
                mass1 = s1.get("precursorList").get("precursor")[0].get("selectedIonList").get("selectedIon")[0].get(
                    "selected ion m/z")
                # mass1 = float(s1.get('params').get('pepmass')[0])
                # charge1 = int(s1.get('params').get('charge').__str__()[0])
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
                        self.spectra_dataset = spectrum01
                    else:
                        self.spectra_dataset = np.vstack((self.spectra_dataset, spectrum01))
                    peakslist1.clear()
                    precursor_feature_list1.clear()
                    ndp_spec_list.clear()
                    j = i * encode_batch

                elif (j + encode_batch + charge_none_record) > self.len:
                    if len(peakslist1) == self.len - j - charge_none_record:
                        tmp_precursor_feature_list1 = np.array(precursor_feature_list1)
                        intensList01 = np.array(peakslist1)

                        # 归一化点积的计算
                        tmp_dplist01 = caculate_nornalization_dp(reference_intensity, ndp_r_spec_list,
                                                                 np.array(peakslist1), np.array(ndp_spec_list))

                        tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
                        spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

                        self.spectra_dataset = np.vstack((self.spectra_dataset, spectrum01))

                        peakslist1.clear()
                        precursor_feature_list1.clear()
                        ndp_spec_list.clear()
                    else:
                        continue

        if len(charge_none_list) > 0:
            np_mr = np.array(charge_none_list)
            df_mr = pd.DataFrame(np_mr, index=None, columns=None)
            df_mr.to_csv(miss_saveName)
            print("Charge Missing Number:{}".format(charge_none_record))
            del charge_none_list

        return self.spectra_dataset

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

def encode_spectra(input,refrence_spectra,miss_record, output):
    """
    :param input: get .mgf file as input
    :param refrence_spectra: get a .mgf file contained 500 spectra as referece spectra from normalized dot product calculation
    :param miss_record: record title of some spectra which loss charge attribute
    :param output: a file for save the final encode information
    :return: None
    """
    if not os.path.exists(input):
        raise RuntimeError("Can not find mgf file: '%s'" % input)
    if str(input).endswith(".mgf"):
        spectra_num = more_itertools.ilen(mgf_read(input, convert_arrays=1))
        mgf_encoder = EncodeDataset(spectra_num)
        vstack_data = mgf_encoder.transform_mgf(input, refrence_spectra, miss_record)
    elif str(input).endswith(".mzML"):
        print(input)
        spectra_num = more_itertools.ilen(mzml_read(input))
        mzml_encoder = EncodeDataset(spectra_num)
        vstack_data = mzml_encoder.transform_mzml(input, refrence_spectra, miss_record)
    np.savetxt(output, vstack_data)
    print("Finish spectra encoding!")
    return vstack_data

if __name__ == '__main__':
    print("test mzml")
    mzml_file = "CHPP_LM3_RP10_1.mzML"
    model = "080802_20_1000_NM500R_model.pkl"
    ref_spectra = "0722_500_rf_spectra.mgf"

    encode_spectra(mzml_file, ref_spectra, miss_record="miss_record.txt", output="0209_test_output.txt")





