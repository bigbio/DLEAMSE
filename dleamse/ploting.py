# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:31:53 2021

@author: xiyang

Drawing
"""
import numpy as np
import matplotlib.pyplot as plt

class Ploting():
    def __init__(self):
        pass
    
    ''' Plot clustering results with only a few clusters '''
    def plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
    
        fig = plt.figure()
        #ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(label[i]),
                     color=plt.cm.Set1(label[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig
    
    ''' Plot clustering results with multiple clusters (the number of clusters exceeds 20)'''
    def multiColorPlotEmbedding(self,data,label,colors,styles1,title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        #print(x_min,x_max)
        #print(data)
        fig = plt.figure(figsize=(25,25),dpi=480)
        #ax = plt.subplot(111)
        print(data.shape[0])
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], styles1[i],
                     color=colors[i],
                     fontdict={'weight': 'bold', 'size': 20})
        plt.xticks([])
        plt.yticks([])
        plt.title(title)
        return fig