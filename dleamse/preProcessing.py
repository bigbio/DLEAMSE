# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:47:49 2021

@author: xiyang


Preprocessing and generating styles
"""

import os
import numpy as np
import collections


class PreProcessing():
    def __init__(self,labelPath,vectorsPath):
        self.label_path = labelPath
        self.vectors_path = vectorsPath
    
    def test(self,value):
        print("testing ",value)
    
    ''' Get all file names in a folder'''
    def file_name(self,file_dir):   
        files1 = []
        for root, dirs, files in os.walk(file_dir):  
            #print(root) #Current directory path
            #print(dirs) #All subdirectories under the current path
            #print(files) #All non-directory sub-files under the current path
            files1=files
        return files1
    
    ''' Preprocess clustering result files and N-dimensional vector files'''
    def preprocessData(self):
        '''
        Correspondence between labels and actual data
        '''
        #labelPath = "C:/Users/xiyang/Desktop/OMEGA/1rank/聚类/demo_spectrum_communities.txt"
        labelPath = self.label_path
        labelData=[]
        with open(labelPath,'r') as f:
            labelData = f.readlines()
        labelMap={}
        for i in range(0,len(labelData)):
            temp = labelData[i]
            labelMap[str(i)] = temp.split("\n")[0].split(",")
        #print(labelMap)
        
        '''
        Extract the N-dimensional vector corresponding to each data used for clustering
        '''
        #files = self.file_name('C:/Users/xiyang/Desktop/OMEGA/1rank/聚类/data')
        files = self.file_name(self.vectors_path)
        #print(files)
        vactors={}
        for i1 in range(0,len(files)):
            #temppath = "C:/Users/xiyang/Desktop/OMEGA/1rank/聚类/data/"+files[i1]
            temppath = self.vectors_path+files[i1]
            vactor=[]
            with open(temppath,'r') as f1:
                vactor = f1.readlines()
            
            vactor1 = []
            for j in range(0,len(vactor)):
                tempV = vactor[j].split("\n")[0].split("\t")[:-1]
                tempV1=[]
                for n in range(0,len(tempV)):
                    tempV1.append(float(tempV[n]))
                vactor1.append(tempV1)
            vactors[files[i1].split(".txt")[0]] = vactor1
        #print(vactors)
        
        return labelMap,vactors
    
    ''' Assemble the tags in the cluster file into a specific format'''
    def assemData(self,labelMap,vactors):
        ''' Assembly label'''
        labels=[]
        data=[]
        for k,v in labelMap.items():
            v1 = v[0]
            if vactors.__contains__(v1):
                temp = vactors[v1]
                for i1 in range(0,len(temp)):
                    data.append(temp[i1])
            for i in range(0,len(v)):
                labels.append(int(k))
        dff = np.array(data)
        #print(dff,dff.shape)
        #print(labels,len(labels))
        return dff,labels,dff.shape[0],dff.shape[1]
    
    ''' Generate the color and mark style of each cluster in the cluster map'''
    def generateColor(self,label):
        # https://www.cnblogs.com/darkknightzh/p/6117528.html
        cnames = {
            'aquamarine':           '#7FFFD4',
            'black':                '#000000',
            'blue':                 '#0000FF',
            'blueviolet':           '#8A2BE2',
            'brown':                '#A52A2A',
            'burlywood':            '#DEB887',
            'cadetblue':            '#5F9EA0',
            'chartreuse':           '#7FFF00',
            'chocolate':            '#D2691E',
            'coral':                '#FF7F50',
            'cornflowerblue':       '#6495ED',
            'crimson':              '#DC143C',
            'cyan':                 '#00FFFF',
            'darkblue':             '#00008B',
            'darkcyan':             '#008B8B',
            'darkgoldenrod':        '#B8860B',
            'darkgray':             '#A9A9A9',
            'darkgreen':            '#006400',
            'darkkhaki':            '#BDB76B',
            'darkmagenta':          '#8B008B',
            'darkolivegreen':       '#556B2F',
            'darkorange':           '#FF8C00',
            'darkorchid':           '#9932CC',
            'darkred':              '#8B0000',
            'darksalmon':           '#E9967A',
            'darkseagreen':         '#8FBC8F',
            'darkslateblue':        '#483D8B',
            'darkslategray':        '#2F4F4F',
            'darkturquoise':        '#00CED1',
            'darkviolet':           '#9400D3',
            'deeppink':             '#FF1493',
            'deepskyblue':          '#00BFFF',
            'dimgray':              '#696969',
            'dodgerblue':           '#1E90FF',
            'firebrick':            '#B22222',
            'forestgreen':          '#228B22',
            'fuchsia':              '#FF00FF',
            'gold':                 '#FFD700',
            'goldenrod':            '#DAA520',
            'gray':                 '#808080',
            'green':                '#008000',
            'greenyellow':          '#ADFF2F',
            'hotpink':              '#FF69B4',
            'indianred':            '#CD5C5C',
            'indigo':               '#4B0082',
            'lawngreen':            '#7CFC00',
            'lightblue':            '#ADD8E6',
            'lightcoral':           '#F08080',
            'lightgreen':           '#90EE90',
            'lightsalmon':          '#FFA07A',
            'lightseagreen':        '#20B2AA',
            'lightskyblue':         '#87CEFA',
            'lightslategray':       '#778899',
            'lime':                 '#00FF00',
            'limegreen':            '#32CD32',
            'magenta':              '#FF00FF',
            'maroon':               '#800000',
            'mediumaquamarine':     '#66CDAA',
            'mediumblue':           '#0000CD',
            'mediumorchid':         '#BA55D3',
            'mediumpurple':         '#9370DB',
            'mediumseagreen':       '#3CB371',
            'mediumslateblue':      '#7B68EE',
            'mediumspringgreen':    '#00FA9A',
            'mediumturquoise':      '#48D1CC',
            'mediumvioletred':      '#C71585',
            'midnightblue':         '#191970',
            'navy':                 '#000080',
            'olive':                '#808000',
            'olivedrab':            '#6B8E23',
            'orange':               '#FFA500',
            'orangered':            '#FF4500',
            'orchid':               '#DA70D6',
            'palegoldenrod':        '#EEE8AA',
            'palegreen':            '#98FB98',
            'paleturquoise':        '#AFEEEE',
            'palevioletred':        '#DB7093',
            'peru':                 '#CD853F',
            'purple':               '#800080',
            'red':                  '#FF0000',
            'rosybrown':            '#BC8F8F',
            'royalblue':            '#4169E1',
            'saddlebrown':          '#8B4513',
            'salmon':               '#FA8072',
            'sandybrown':           '#FAA460',
            'seagreen':             '#2E8B57',
            'sienna':               '#A0522D',
            'skyblue':              '#87CEEB',
            'slateblue':            '#6A5ACD',
            'slategray':            '#708090',
            'springgreen':          '#00FF7F',
            'steelblue':            '#4682B4',
            'tan':                  '#D2B48C',
            'teal':                 '#008080',
            'tomato':               '#FF6347',
            'turquoise':            '#40E0D0',
            'violet':               '#EE82EE',
            'wheat':                '#F5DEB3',
            'yellow':               '#FFFF00',
            'yellowgreen':          '#9ACD32'
        }
        
        '''
        styles=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',\
                't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', \
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','1','2','3','4','5','6','7','8','9','0'\
               '.','>','<','-','|','o',',','v','^','*','+',]
        '''
            
        styles = [x for x in range(0,len(label))]
        
        colorskey=[]
        colors=[]
        styles1=[]
        count=0
        count1=0
        clabel = collections.Counter(label)
        #print(clabel)
        
        for k,v in cnames.items():
            colorskey.append(v)
            
        for i in range(0,len(label)):
            number = clabel[i]
            #colors
            if count == len(colorskey):
                count=0
                for j in range(0,number):
                    colors.append(colorskey[count])
            else:
                for j in range(0,number):
                    colors.append(colorskey[count])
                count=count+1
                
            #styles
            if count1 == len(styles):
                count1=0
                for j1 in range(0,number):
                    if number>20:
                        styles1.append(number)
                    else:
                        styles1.append(styles[count1])
            else:
                for j1 in range(0,number):
                    if number>20:
                        styles1.append(number)
                    else:
                        styles1.append(styles[count1])
                count1=count1+1
            
        
        #print("colors len:",len(colors),len(styles1))
        return colors,styles1
