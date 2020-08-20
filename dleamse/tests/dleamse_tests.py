import os
import sys
 
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
parent_dir = os.path.dirname(__file__) + "/.."
sys.path.append(parent_dir)
sys.path.append("..") #./dleamse/
print(sys.path)

abspath_dleamse = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/"
print(abspath_dleamse)

from dleamse_encode_and_embed import encode_and_embed_spectra
from dleamse_encode_and_embed import SiameseNetwork2
from dleamse_faiss_index_writer import FaissWriteIndex
from dleamse_faiss_index_search import FaissIndexSearch
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import logging
import click
import warnings

from click.testing import CliRunner

#from dleamse.mslookup import cli
from mslookup import cli

def embeded_db_spectra():
    runner = CliRunner()
    result = runner.invoke(cli,
                           ['embed-ms-file', '-m', abspath_dleamse+'dleamse_model_references/080802_20_1000_NM500R_model.pkl', '-r', abspath_dleamse+'dleamse_model_references/0722_500_rf_spectra.mgf', '-i', abspath_dleamse+'testdata/PXD015890_114263_ArchiveSpectrum.json',
                            '-p', 'PXD015890'])
    """
    python mslookup.py embed-ms-file -i testdata/PXD015890_114263_ArchiveSpectrum.json -p PXD015890
    """
    print(result)
    print(result.output)
    print(result.exit_code)
    assert result.exit_code == 0


def make_db():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['make-index', '-d', abspath_dleamse+'testdata/db.csv',
                          '-e', abspath_dleamse+'testdata/', '-o', abspath_dleamse+'testdata/db.index'])
  """
  python mslookup.py make-index -d testdata/db.csv -e testdata/ -o testdata/db.index
  """
  print(result)
  print(result.output)
  print(result.exit_code)
  assert result.exit_code == 0

def embeded_query_spectra():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['embed-ms-file', '-m', abspath_dleamse+'dleamse_model_references/080802_20_1000_NM500R_model.pkl', '-r',abspath_dleamse+'dleamse_model_references/0722_500_rf_spectra.mgf', '-i', abspath_dleamse+'testdata/query.json',
                          '-p', 'PXD015890'])
  """
  python mslookup.py embed-ms-file -i testdata/query.json -p PXD015890
  """
  print(result)
  print(result.output)
  print(result.exit_code)
  assert result.exit_code == 0

def clean_db():
  #os.remove(abspath_dleamse+"testdata/PXD015890_114263_ArchiveSpectrum_encoded.npy")
  os.remove(abspath_dleamse+"testdata/PXD015890_114263_ArchiveSpectrum_ids_usi.txt")
  os.remove(abspath_dleamse+"testdata/db.index")
  #os.remove(abspath_dleamse+"testdata/usi_db.csv") #No such file was generated
  os.remove(abspath_dleamse+"testdata/db_ids_usi.csv")
  #os.remove(abspath_dleamse+"testdata/query_encoded.npy")
  os.remove(abspath_dleamse+"testdata/query_ids_usi.txt")


def search_spectra():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['range-search', '-i', abspath_dleamse+'testdata/db.index',
                          '-u', abspath_dleamse+'testdata/db_ids_usi.csv', '-n', 100,'-e', abspath_dleamse+'testdata/query_embedded.txt', '-o', abspath_dleamse+'testdata/minor.csv', '-ut', 0.099, '-lt', 0.0])
  """
  result = runner.invoke(cli,
                         ['range-search', '-i', 'testdata/db.index',
                          '-u', 'testdata/db_ids_usi.csv', '-e', 'testdata/query_encoded_embedded.txt', '-o', 'testadata/minor.csv', '-ut', '0.099', '-lt', '0.07'])

                          PXD015890_114263_ArchiveSpectrum_embedded.txt
  python mslookup.py range-search -i testdata/db.index -u testdata/db_ids_usi.csv -e testdata/query_embedded.txt -lt 0.0 -ut 0.099 -o testdata/minor.json
  """
  print(result)
  print(result.output)
  print(result.exit_code)
  assert result.exit_code == 0

if __name__ == '__main__':
    embeded_db_spectra()
    make_db()
    embeded_query_spectra()
    search_spectra()
    clean_db()

