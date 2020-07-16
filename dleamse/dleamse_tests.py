import os

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
                           ['embed-ms-file', '-i', 'testdata/PXD015890_114263_ArchiveSpectrum.json',
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
                         ['make-index', '-d', 'testdata/db.csv',
                          '-e', 'testdata/', '-o', 'testdata/db.index'])
  """
  python mslookup.py make-index -d testdata/db.csv -e testdata/ -o testdata/db.index
  """
  assert result.exit_code == 0

def embeded_query_spectra():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['embed-ms-file', '-i', 'testdata/query.json',
                          '-p', 'PXD015890'])
  """
  python mslookup.py embed-ms-file -i testdata/query.json -p PXD015890
  """
  print(result)
  print(result.output)
  print(result.exit_code)
  assert result.exit_code == 0

def clean_db():
  os.remove("testdata/PXD015890_114263_ArchiveSpectrum_encoded.npy")
  os.remove("testdata/PXD015890_114263_ArchiveSpectrum_ids_usi.txt")
  os.remove("testdata/db.index")
  #os.remove("testdata/usi_db.csv") #No such file was generated
  os.remove("testdata/db_ids_usi.csv")
  os.remove("testdata/query_encoded.npy")
  os.remove("testdata/query_ids_usi.txt")


def search_spectra():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['range-search', '-i', 'testdata/db.index',
                          '-u', 'testdata/db_ids_usi.csv', '-n', 100,'-e', 'testdata/query_embedded.txt', '-o', 'testdata/minor.csv', '-ut', 0.099, '-lt', 0.0])
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

