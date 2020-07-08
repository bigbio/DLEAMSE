import os

from click.testing import CliRunner

from dleamse.mslookup import cli

def embeded_db_spectra():
    runner = CliRunner()
    result = runner.invoke(cli,
                           ['embed-ms-file', '-i', 'testdata/PXD015890_114263_ArchiveSpectrum.json',
                            '-p', 'PXD015890'])
    assert result.exit_code == 0


def make_db():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['make-index', '-d', 'testdata/db.csv',
                          '-e', 'testdata/', '-o', 'testdata/db.index'])
  assert result.exit_code == 0

def embeded_query_spectra():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['embed-ms-file', '-i', 'testdata/query.json',
                          '-p', 'PXD015890'])
  assert result.exit_code == 0

def clean_db():
  os.remove("testdata/PXD015890_114263_ArchiveSpectrum_encoded.npy")
  os.remove("testdata/PXD015890_114263_ArchiveSpectrum_ids_usi.txt")
  os.remove("testdata/db.index")
  os.remove("testdata/usi_db.csv")
  os.remove("testdata/db_ids_usi.csv")
  os.remove("testdata/query_encoded.npy")
  os.remove("testdata/query_ids_usi.txt")


def search_spectra():
  runner = CliRunner()
  result = runner.invoke(cli,
                         ['range-search', '-i', 'testdata/db.index',
                          '-u', 'testdata/db_ids_usi.csv', '-e', 'testdata/query_encoded_embedded.txt', '-o', 'testadata/minor.csv', '-ut', '0.099', '-lt', '0.07'])

  assert result.exit_code == 0

if __name__ == '__main__':
    embeded_db_spectra()
    make_db()
    embeded_query_spectra()
    search_spectra()
    clean_db()

