# -*- coding:utf-8 -*-
"""
This is a search program!
Create by qincy, April 17,2019
"""

#将数据进行嵌入
import math
import os
import logging

import numpy
import seaborn as sns
import torch
from pyteomics.mgf import read
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
from numpy import concatenate

import matplotlib as mpl
import matplotlib.pyplot as plt

class SiameseNetwork2(nn.Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()

        # self.fc1_1 = nn.Linear(34, 32)
        self.fc1_1 = nn.Linear(34, 16)
        # self.fc1_1 = nn.Linear(61, 32)
        self.fc1_2 = nn.Linear(16, 5)
        # self.fc1_2 = nn.Linear(32, 5)

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
        refSpecInfo = refSpecInfo.view(refSpecInfo.size(0), -1)  # 改变数据的形状，-1表示不确定，视情况而定

        output = torch.cat((preInfo, fragInfo, refSpecInfo), 1)
        # output = self.dropout(output)
        output = self.fc2(output)
        output = F.selu(output)
        return output

    def forward(self, spectrum01, spectrum02):

        spectrum01 = spectrum01.reshape(spectrum01.shape[0], 1, spectrum01.shape[1])
        spectrum02 = spectrum02.reshape(spectrum02.shape[0], 1, spectrum02.shape[1])

        input1_1 = spectrum01[:, :, :100]
        input1_2 = spectrum01[:, :, 100:2549]
        input1_3 = spectrum01[:, :, 2549:]

        input2_1 = spectrum02[:, :, :100]
        input2_2 = spectrum02[:, :, 100:2549]
        input2_3 = spectrum02[:, :, 2549:]

        output01 = self.forward_once(input1_3, input1_2, input1_1)
        output02 = self.forward_once(input2_3, input2_2, input2_1)

        return output01, output02

class RawDataSet01():

    def __init__(self, mgf_file, csv_file, saveName):
        if not os.path.exists(mgf_file):
            raise RuntimeError("Can not find mgf file: '%s'" % mgf_file)
        if not os.path.exists(csv_file):
            raise RuntimeError("Can not find csv file: '%s'" % csv_file)
        self.MGF = {}
        self.mgf_dataset = []
        self.load_file(mgf_file, csv_file)
        self.transform(saveName)
        # pd.DataFrame(self.mgf_dataset).to_csv("../Data/train.data", header=None, index=None)

    def load_file(self, mgf_path, csv_path):
        print('Start to load file data...')
        info = pd.read_csv(csv_path, header=None)
        self.pairs_num = info.shape[0]
        self.spectrum1 = info[0].tolist()
        # self.spectrum2 = info[1].tolist()
        # self.label = info[2].tolist()
        for mgf in read(mgf_path, convert_arrays=1):
            self.MGF[mgf.get('params').get('title').replace('id=', '')] = mgf
        print('Finish to load data...')

    def transform(self, saveName):
        print('Start to calculate data set...')

        global spectrum_dict
        spectrum_dict = {}
        #五百个参考的谱图
        # reference_spectra = read("../SpectraPairsData/0715_50_rf_spectra.mgf", convert_arrays=1)
        # reference_spectra = read("../SpectraPairsData/0628_100_rf_spectra.mgf", convert_arrays=1)
        reference_spectra = read("../SpectraPairsData/0722_500_rf_spectra.mgf", convert_arrays=1)
        reference_intensity = np.array([self.bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in reference_spectra])
        # 先将500个参考谱图的点积结果计算出来
        # ndp_r_spec_list = np.zeros(50)
        # ndp_r_spec_list = np.zeros(100)
        ndp_r_spec_list = np.zeros(500)

        # for x in range(50):
        # for x in range(100):
        for x in range(500):
            ndp_r_spec = np.math.sqrt(np.dot(reference_intensity[x], reference_intensity[x]))
            ndp_r_spec_list[x] = ndp_r_spec

        peakslist1, precursor_feature_list1 = [], []
        ndp_spec_list1 = []
        i, j, k = 0, 0, 0

        for s1 in self.spectrum1:
            s1 = self.MGF[s1]

            bin_s1 = self.bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
            ndp_spec1 = np.math.sqrt(np.dot(bin_s1, bin_s1))
            peakslist1.append(bin_s1)
            ndp_spec_list1.append(ndp_spec1)
            mass1 = float(s1.get('params').get('pepmass')[0])
            charge1 = int(s1.get('params').get('charge').__str__()[0])
            precursor_feature1 = np.concatenate((self.gray_code(mass1), self.charge_to_one_hot(charge1)))
            precursor_feature_list1.append(precursor_feature1)

            if len(peakslist1) == 500:
                i += 1

                tmp_precursor_feature_list1 = np.array(precursor_feature_list1)

                intensList01 = np.array(peakslist1)

                # 归一化点积的计算
                tmp_dplist01 = self.caculate_nornalization_dp(reference_intensity, ndp_r_spec_list, np.array(peakslist1), np.array(ndp_spec_list1))

                tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
                spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)

                self.mgf_dataset = spectrum01

                df = pd.DataFrame(self.mgf_dataset)
                df.to_csv(saveName, mode="a+", header=False, index=False)

                del self.mgf_dataset
                peakslist1.clear()
                precursor_feature_list1.clear()
                ndp_spec_list1.clear()

                j = i * 500

            elif (j+500) > self.pairs_num:
                num = self.pairs_num - j
                k += 1
                if num == k:

                    tmp_precursor_feature_list1 = np.array(precursor_feature_list1)

                    intensList01 = np.array(peakslist1)

                    # 归一化点积的计算
                    tmp_dplist01 = self.caculate_nornalization_dp(reference_intensity, ndp_r_spec_list,
                                                                  np.array(peakslist1), np.array(ndp_spec_list1))

                    tmp01 = concatenate((tmp_dplist01, intensList01), axis=1)
                    spectrum01 = concatenate((tmp01, tmp_precursor_feature_list1), axis=1)
                    self.mgf_dataset = spectrum01

                    df = pd.DataFrame(self.mgf_dataset)
                    df.to_csv(saveName, mode="a+", header=False, index=False)

                    del self.mgf_dataset
                    peakslist1.clear()
                    precursor_feature_list1.clear()
                    ndp_spec_list1.clear()
                else:
                    continue
            else:
                continue

        del self.MGF
        print('Finish to calculate data set...')

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
        ndp_r_spec_list = ndp_r_spec_list.reshape(ndp_r_spec_list.shape[0],1)
        ndp_bin_sp = ndp_bin_sp.reshape(ndp_bin_sp.shape[0], 1)
        tmp_dp_list = np.dot(bin_spectra, np.transpose(reference))
        dvi = np.dot(ndp_bin_sp, np.transpose(ndp_r_spec_list))
        result = tmp_dp_list / dvi
        return result

class Dataset_RawDataset(data.dataset.Dataset):
    def __init__(self, data_file):
        if not os.path.exists(data_file):
            raise RuntimeError("Can not find mgf file: '%s'" % data_file)
        self.mgf_dataset = pd.read_csv(data_file, error_bad_lines=False, header=None, index_col=None).values
        print(self.mgf_dataset.shape)

    def __getitem__(self, item):
        return self.mgf_dataset[item]

    def __len__(self):
        return len(self.mgf_dataset)

class NewMGFDataSet(data.dataset.Dataset):

    def __init__(self, mgf_file, csv_file, reference_spectra_number=500):
        if not os.path.exists(mgf_file):
            raise RuntimeError("Can not find mgf file: '%s'" % mgf_file)
        if not os.path.exists(csv_file):
            raise RuntimeError("Can not find csv file: '%s'" % csv_file)
        self.MGF = {}
        self.mgf_dataset = []
        self.load_file(mgf_file, csv_file)
        self.transform(reference_spectra_number)
        # print(self.mgf_dataset)info

    def load_file(self, mgf_path, csv_path):
        print('Start to load file data...')
        info = pd.read_csv(csv_path, sep=",", header=None, index_col=None)
        self.spectrum1 = info[0].tolist()
        for mgf in read(mgf_path, convert_arrays=1):
            #只是选出scan作为id
            title = mgf.get('params').get('title').replace('id=', '')
            # if len(title.split(",")) > 1:
            #     NativeID = title.strip(" ").split(",")[1]
            #     id = NativeID.strip(" ").split(":")[1]
            #     final_id = id.strip("\"")
            #     self.MGF[final_id] = mgf
            # else:
            self.MGF[title] = mgf

        print('Finish to load data...')

    def transform(self, reference_spectra_number):
        print('Start to calculate data set...')

        global spectrum_dict
        spectrum_dict = {}
        #五百个参考的谱图
        # rfDataFrame = pd.read_csv("../Data/500RfSpectraBinData.csv", header=None, index_col=None)
        # reference_intensity = rfDataFrame.values
        rfData = read("../SpectraPairsData/gleams_reference_spectra.mgf", convert_arrays=1)
        reference_intensity = np.array([self.bin_spectrum(r.get('m/z array'), r.get('intensity array')) for r in rfData])

        peakslist1, precursor_feature_list1 = [], []
        for s1 in self.spectrum1:
            s1 = self.MGF[s1]
            bin_s1 = self.bin_spectrum(s1.get('m/z array'), s1.get('intensity array'))
            peakslist1.append(bin_s1)

            mass1 = float(s1.get('params').get('pepmass')[0])
            charge1 = int(s1.get('params').get('charge').__str__()[0])
            mz1 = mass1 / charge1
            precursor_feature1 = np.concatenate(
                (self.gray_code(mass1), self.gray_code(mz1), self.charge_to_one_hot(charge1)))
            precursor_feature_list1.append(precursor_feature1)

        intensList01 = np.array(peakslist1)
        refMatrix = np.transpose(reference_intensity)
        DPList01 = np.dot(intensList01, refMatrix)

        precursor_feature_list1 = np.array(precursor_feature_list1)

        tmp01 = np.concatenate((DPList01, intensList01), axis=1)
        spectrum01 = np.concatenate((tmp01, precursor_feature_list1), axis=1)

        self.mgf_dataset = spectrum01

        del self.MGF
        print('Finish to calculate data set...')

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
        # assert num.is_integer(), 'Parameter "num" must be integer'
        number = np.int(number)
        # we need 27-bit "Gray Code"
        bit = 27
        shift = 1
        gray_code = np.binary_repr(np.bitwise_xor(number, np.right_shift(number, shift)), bit)
        # print(type(gray_code))
        return np.asarray(' '.join(gray_code).split(), dtype=float)

    def charge_to_one_hot(self, c: int):
        """
        encode charge with one-hot format for 1-7
        :param c:
        :return:
        """
        maximum_charge = 7
        charge = np.zeros(maximum_charge, dtype=float)
        # if charge bigger than 7, use 7 instead
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
        key = mz_array.__str__()
        if key in spectrum_dict.keys():  # use cache just take 4s
            # if False: use the old one may take 7s for 50
            return spectrum_dict[key]
        else:
            nbins = int(float(max_mz - min_mz) / float(bin_size)) + 1
            results = np.zeros(nbins)
            for index in range(len(mz_array)):
                mz = mz_array[index]
                intensity = intensity_array[index]
                #进行平方根变换
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
                # results[bin_index] = intensity

            intensity_sum = results.sum()
            if intensity_sum > 0:
                results /= intensity_sum
                spectrum_dict[key] = results
            else:
                logging.debug('zero intensity found')
        return results

    def __getitem__(self, item):
        return self.mgf_dataset[item]

    def __len__(self):
        return len(self.mgf_dataset)

class EmbedDataSet():
    def __init__(self):
        self.out_list = []

    def createSpecPairsFromMgfFile(self, mgfFile, storeCsvFile):
        print("Start Embedding ...")
        SpecPair = {}

        data = read(mgfFile, convert_arrays=1)

        for dt in data:
            title = dt.get('params').get('title').replace('id=', '')
            # print(title)
            # t_01 = str(title).replace(" N", "N").replace(" ", "_")
            # if len(title.split(",")) > 1:
            #     NativeID = title.strip(" ").split(",")[1]
            #     id = NativeID.strip(" ").split(":")[1]
            #     final_id = id.strip("\"")
            #     SpecPair["spectrum"] = final_id
            # else:
            #     SpecPair["spectrum"] = title
            SpecPair["spectrum"] = title

            SpecPair["charge"] = int(dt.get('params').get('charge').__str__()[0])
            SpecPair["pepmass"] = float(dt.get('params').get('pepmass')[0])

            df = pd.DataFrame(SpecPair, columns=["spectrum", "charge", "pepmass"], index=[0])
            df.to_csv(storeCsvFile, mode="a+", header=False, index=False)
            # del df
            SpecPair.clear()

    def embedding_dataset(self, model, raw_data, mgfFile, storeCsvFile, storeEmbedFile):

        batch = 1
        # net = torch.load("../SpectraPairsData/0423_150_128_01_model.pkl", map_location='cpu')
        net = torch.load(model, map_location='cpu')

        #生成csv文件，在存在的情况下不需要执行
        # self.createSpecPairsFromMgfFile(mgfFile, storeCsvFile)
        #生成RAW文件，在存在的情况下不需要重复执行
        # RawDataSet01(mgfFile, storeCsvFile, raw_data)

        dataset = Dataset_RawDataset(raw_data)
        dataloader = data.DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=1)

        df = pd.read_csv(storeCsvFile, header=None, index_col=None)

        for j, test_data in enumerate(dataloader, 0):
            #todo:放spectrum_title在文件的前方

            spectrum01 = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])

            # input1_1 = spectrum01[:, :, :50]
            # input1_2 = spectrum01[:, :, 50:2499]
            # input1_3 = spectrum01[:, :, 2499:]
            input1_1 = spectrum01[:, :, :100]
            input1_2 = spectrum01[:, :, 100:2549]
            input1_3 = spectrum01[:, :, 2549:]
            # input1_1 = spectrum01[:, :, :500]
            # input1_2 = spectrum01[:, :, 500:2949]
            # input1_3 = spectrum01[:, :, 2949:]

            output01 = net.forward_once(input1_3, input1_2, input1_1)

            list1, list2, list_all = [], [], []

            list1.append(df[0][j])
            list1.append(df[1][j])
            list1.append(df[2][j])
            list1.append(output01)

            out1 = output01.detach().numpy()[0]
            self.out_list.append(out1)

        np.savetxt(storeEmbedFile, self.out_list)
        self.out_list.clear()

def All_Logistic_AUC(X, y):
    # data = pd.read_csv("./data/test_auc_data_02.txt", header=None, index_col=None)
    X, y = np.array(X), np.array(y)
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)

    # y_one_hot = label_binarize(y_test, np.arange(2))  # 装换成类似二进制的编码

    alpha = np.logspace(-2, 2, 20)  # 设置超参数范围
    model = LogisticRegressionCV(Cs=alpha, cv=3, penalty='l2')  # 使用L2正则化
    model.fit(x_train, y_train)

    # print('超参数：', model.C_)
    # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]
    y_score = model.decision_function(X)#.predict_proba(x_test)
    # print(y_score)
    # 1、调用函数计算micro类型的AUC
    # print('调用函数auc：', metrics.roc_auc_score(y_one_hot, y_score, average='micro'))
    # 2、手动计算micro类型的AUC
    # 首先将矩阵y_one_hot和y_score展开，然后计算假正例率FPR和真正例率TPR
    fpr, tpr, thresholds = metrics.roc_curve(y, y_score)
    auc = metrics.auc(fpr, tpr)
    # print('手动计算auc：', auc)

    return fpr, tpr, auc

def show_test01_roc(fpr_list, tpr_list, auc_list, saveName):
    # 绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    # FPR就是横坐标,TPR就是纵坐标
    plt.figure(figsize=(10, 10))
    # plt.subplot(2, 1, 1)
    color_list = ['darkkhaki', 'cornflowerblue', 'seagreen', 'steelblue', 'dimgrey', 'red', 'darkorange']
    title_list = ['Pearson\'s', 'Spearman\'', 'NDP\'s', 'DP\'s', 'MSE\'s', 'Model\'s', 'Gleams\'s']

    # line_list = ['--', ]
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i], lw=2, color=color_list[i], alpha=0.7, label=u'%s AUC=%.3f' % (title_list[i],auc_list[i]))
        plt.plot((0, 1), (0, 1), lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'Logistic: ROC and AUC', fontsize=17)
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(len(loss)), loss, 'r')
    # plt.title('Loss for Iteration: %s Batch_size: %s' % (len(loss), batch_size))
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    plt.savefig('./data/auc/' + saveName + '.jpg')
    # plt.show()
    plt.show()

def draw_auc_plot(auc_data, plot_name):
    data = pd.read_csv(auc_data, sep="\t", header=0, index_col=None)
    y = data["label"]
    X_pearson = data["pearson"]
    # tmp_df = df[["Spectrum_Title", "best_matched_spectrum_title", "pearson", "spearman", "normalized_dot_score", "dot_score", "mean_squared_error"]]
    head_list = ["pearson", "spearman", "normalized_dot_score", "dot_score", "mean_squared_error", "m_eu_dis","g_eu_dis"]
    fpr_list, tpr_list, roc_auc_list = [], [], []
    for title in head_list:
        X = data[title]
        fpr, tpr, roc_auc = All_Logistic_AUC(X, y)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
    show_test01_roc(fpr_list, tpr_list, roc_auc_list, plot_name)

def createMouseAUCData(all_model_data, all_model_auc_data, version):
    if version == 1:
        df = pd.read_csv("./data/Mouse/compare_result/0710_414_hits.txt", sep="\t", header=0, index_col=None)
        gleams_df = pd.read_csv("./data/Mouse/Gleams/0710_414_gleams_hits.txt", sep="\t", header=0, index_col=None)
    else:
        df = pd.read_csv("./data/Mouse/compare_result/0710_all_hits.txt", sep="\t", header=0, index_col=None)
        gleams_df = pd.read_csv("./data/Mouse/Gleams/0710_all_gleams_hits.txt", sep="\t", header=0, index_col=None)

    model_df = pd.read_csv(all_model_data, sep="\t", header=0, index_col=None)
    model_dis = model_df["m_eu_dis"]
    gleams_dis = gleams_df["g_eu_dis"]
    label_df = df["label"]
    tmp_df = df[["Spectrum_Title", "best_matched_spectrum_title", "pearson", "spearman", "normalized_dot_score", "dot_score", "mean_squared_error"]]

    save_df = pd.concat([tmp_df, model_dis, gleams_dis, label_df], axis=1)
    pd.DataFrame(save_df).to_csv(all_model_auc_data, sep="\t", header=True, index=None)
def CreateModelMousehitsFiles(yeast_sigma_embed, sigma_embed, all_model_data, version):
    ups_yeast_scans_df = pd.read_csv("./data/Mouse/Model/0711_062507_15_256_NM_NR_04195_SCAN.csv", header=None,index_col=None)
    ups_yeast_embed_df = numpy.loadtxt(yeast_sigma_embed)

    ups_scans_df = pd.read_csv("./data/Mouse/Model/0711_062507_15_256_NM_NR_04197_SCAN.csv", header=None, index_col=None)
    ups_embed_df = numpy.loadtxt(sigma_embed)

    dim_data01 = ups_yeast_embed_df.reshape(ups_yeast_embed_df.shape[0], 1, ups_yeast_embed_df.shape[1])
    dim_data02 = ups_embed_df.reshape(ups_embed_df.shape[0], 1, ups_embed_df.shape[1])

    if version == 1:
        all_hits_df = pd.read_csv("./data/Mouse/compare_result/0710_414_hits.txt", sep="\t",header=0, index_col=None)
    else:
        all_hits_df = pd.read_csv("./data/Mouse/compare_result/0710_all_hits.txt", sep="\t", header=0, index_col=None)
    # print(all_hits_df)

    spectrum = all_hits_df["Spectrum_Title"].values.tolist()
    best_matched_spectrum = all_hits_df["best_matched_spectrum_title"].values.tolist()
    label = all_hits_df["label"].values.tolist()

    model_list = []
    for i in range(len(spectrum)):
        tmp_list = []
        tmp_spec_01 = spectrum[i].split(",")[1].split(":")[1].strip("\"").split("_")[2].split("=")[1]
        tmp_matched_spec_02 = best_matched_spectrum[i].split(",")[1].split(":")[1].strip("\"").split("_")[2].split("=")[1]
        for j in range(ups_yeast_scans_df.shape[0]):
            spec_01 = ups_yeast_scans_df[0][j].split(",")[1].split(":")[1].strip("\"").split(" ")[2].split("=")[1]
            if tmp_spec_01 == spec_01:
                tmp_list.append(spectrum[i])
                dim_d1 = torch.from_numpy(dim_data01[j])
                for k in range(ups_scans_df.shape[0]):
                    matched_spec_02 = ups_scans_df[0][k].split(",")[1].split(":")[1].strip("\"").split(" ")[2].split("=")[1]
                    if tmp_matched_spec_02 == matched_spec_02:
                        tmp_list.append(best_matched_spectrum[i])
                        dim_d2 = torch.from_numpy(dim_data02[k])
                        euclidean_distance = F.pairwise_distance(dim_d1, dim_d2)
                        euclidean_dis = euclidean_distance.detach().numpy()[0]
                        tmp_list.append(euclidean_dis)
                        tmp_list.append(label[i])
                        model_list.append(tmp_list)
                        break
                break

    final_data = pd.DataFrame(numpy.array(model_list), columns=["Spectrum_Title","best_matched_spectrum_title", "m_eu_dis", "label"])
    final_data.to_csv(all_model_data, sep="\t", header=True, index=False)
def ExecuteModelCompare_Mouse(version, model, model_data_version):

    # .embed文件需要每次都生成
    storeEmbedFile01 = "./data/Mouse/compare_result/" + model_data_version + "_04195_SCAN_embed.txt"
    storeEmbedFile02 = "./data/Mouse/compare_result/" + model_data_version + "_04197_SCAN_embed.txt"

    # .raw文件是存放经过编码的谱图数据的，因此在不改变数据原始编码维度的情况下 可以再次使用
    # .csv文件对应.raw文件的谱图顺序，可以重复使用
    date_version = "080702_10_1000_NM100R"


    spec01_mgfFile = "./data/Mouse/db_search_result/0710_04195_SCAN.mgf"
    mgf_rawFile01 = "./data/Mouse/Model/" + date_version + "_04195_SCAN.raw"
    spec01_csvFile = "./data/Mouse/Model/" + date_version + "_04195_SCAN.csv"
    spec02_mgfFile = "./data/Mouse/db_search_result/0710_04197_SCAN.mgf"
    mgf_rawFile02 = "./data/Mouse/Model/" + date_version + "_04197_SCAN.raw"
    spec02_csvFile = "./data/Mouse/Model/" + date_version + "_04197_SCAN.csv"

    embedder01 = EmbedDataSet()
    embedder01.embedding_dataset(model, mgf_rawFile01, spec01_mgfFile, spec01_csvFile, storeEmbedFile01)
    embedder02 = EmbedDataSet()
    embedder02.embedding_dataset(model, mgf_rawFile02, spec02_mgfFile, spec02_csvFile, storeEmbedFile02)

    if version == 1:
        all_model_seaborn_data = "./data/Mouse/Model/414_model_hits_" + model_data_version + ".txt"
        all_model_auc_data = "./data/Mouse/Model/414_hits_auc_data_" + model_data_version + ".txt"
        plot_name = model_data_version + "_Mouse_414_logistic"
    else:
        all_model_seaborn_data = "./data/Mouse/Model/all_model_hits_" + model_data_version + ".txt"
        all_model_auc_data = "./data/Mouse/Model/all_hits_auc_data_" + model_data_version + ".txt"
        plot_name = model_data_version + "_Mouse_all_logistic"

    CreateModelMousehitsFiles(storeEmbedFile01, storeEmbedFile02, all_model_seaborn_data, version)
    createMouseAUCData(all_model_seaborn_data, all_model_auc_data, version)
    draw_auc_plot(all_model_auc_data, plot_name)

if __name__ == '__main__':
    # 1代表正负样本一致
    # 0代表全部样本数据

    model = "../SpectraPairsData/081501_15_1000_NM100R_model.pkl"
    model_data_version = "081501_15_1000_NM100R"

    ExecuteModelCompare_Mouse(1, model, model_data_version)
