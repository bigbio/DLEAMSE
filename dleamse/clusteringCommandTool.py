# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:27:21 2021

@author: xiyang
"""

"""
************************************************************************
A command line tool for plotting clustering results of DLEAMSE using t-sne
"""

import logging
import os
import click
from time import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

''' Custom class '''
from preProcessing import PreProcessing
from ploting import Ploting

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """This is the main tool that give access to all commands and options provided by the mslookup and dleamse algorithms"""


class AppConfigException(object):
    def __init__(self, value):
        super(AppConfigException, self).__init__(value)


@click.command('clustering_result_plotter', short_help='Plot the results of clustering')
@click.option('--vector_directory', '-v', help='The file directory that saves the clustering results. \
              Note: Only the directory containing the clustering result files can be entered, \
                  and there are only clustering result files in the directory.(For example: /home/data/)', required=True)
@click.option('--clustering_label_file', '-c', help="Clustering result label file", required=True)
@click.option('--image_save_path', '-i', help="The path where the image is saved. You must give the file name and format of the picture to be saved.(For example: /home/t-sne.jpg)", required=True)
@click.option('--img_title', '-it', help='Name of cluster graph', default="Clustering result graph")
@click.pass_context 
def clustering_result_plotter(ctx,vector_directory,clustering_label_file,image_save_path,img_title):
    
    if vector_directory.split("/")[-1] != "/":
        vector_directory = vector_directory+"/"
        print("end with '/' : ",vector_directory)
    else:
        pass

    pp = PreProcessing(clustering_label_file,vector_directory)
    draw = Ploting()
    labelMap,vactors = pp.preprocessData()
    data, label, n_samples, n_features = pp.assemData(labelMap,vactors)
    
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0,perplexity=6.0)
    #https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
    #https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/manifold/_t_sne.py#L480
    #t0 = time()
    result = tsne.fit_transform(data)
    colors,styles1 = pp.generateColor(label)
    #fig = draw.multiColorPlotEmbedding(result, label,colors,styles1,'t-SNE embedding of the digits (time %.2fs)'% (time() - t0))
    fig = draw.multiColorPlotEmbedding(result, label,colors,styles1,img_title)
    plt.savefig(image_save_path, dpi = 480)
    #plt.show(fig)
  

cli.add_command(clustering_result_plotter)

if __name__ == "__main__":
    cli()