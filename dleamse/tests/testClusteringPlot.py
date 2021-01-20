# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:57:02 2021

@author: xiyang
"""

import os
import sys
 
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
parent_dir = os.path.dirname(__file__) + "/.."
sys.path.append(parent_dir)
sys.path.append("..") #./dleamse/
print(sys.path)

abspath_dleamse = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/"

print("abspath_dleamse:", abspath_dleamse)


from click.testing import CliRunner
from clusteringCommandTool import cli

def clustering_result_plotter():
    runner = CliRunner()
    result = runner.invoke(cli,
                           ['clustering_result_plotter', '-v', abspath_dleamse+'/testdata/32-D_vectors/', '-c', abspath_dleamse+'/testdata/clustering_label.txt', '-i', abspath_dleamse+'/testdata/test.png',
                            '-it', "Clustering result graph"])
    """
    python clusteringCommandTool.py clustering_result_plotter -v ./32-D_vectors/ -c ./clustering_label.txt -i ./test.png -it "Clustering result graph test"
    """
    print(result)
    print(result.output)
    print(result.exit_code)
    assert result.exit_code == 0
    

def clean():
  os.remove(abspath_dleamse+"/testdata/test.png")


if __name__ == '__main__':
    clustering_result_plotter()
    clean()