import logging
import os

import click

import warnings

#from dleamse.dleamse_encode_and_embed import encode_and_embed_spectra
#from dleamse.dleamse_encode_and_embed import SiameseNetwork2
#from dleamse.dleamse_faiss_index_writer import FaissWriteIndex
#from dleamse.dleamse_faiss_index_search import FaissIndexSearch

from dleamse_encode_and_embed import encode_and_embed_spectra
from dleamse_faiss_index_writer import FaissWriteIndex
from dleamse_faiss_index_search import FaissIndexSearch
from dleamse_encode_and_embed import SiameseNetwork2
from numba.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

DEFAULT_IVF_NLIST = 100

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
  """This is the main tool that give access to all commands and options provided by the mslookup and dleamse algorithms"""


class AppConfigException(object):

  def __init__(self, value):
    super(AppConfigException, self).__init__(value)


@click.command('embed-ms-file',
               short_help='Commandline to encode and embed every MS/MS spectrum in a file into a 32 features vector')
@click.option('--model', '-m', help='Input embedder model file',
              default="./dleamse_model_references/080802_20_1000_NM500R_model.pkl")
@click.option('--ref_spectra', '-r', help='Input 500 reference spectra file',
              default="./dleamse_model_references/0722_500_rf_spectra.mgf")
@click.option('--project_accession', '-p', help='ProteomeXchange dataset accession', default="Project_ID")
@click.option('--input_file', '-i', help='Input MS File (supported: mzML, MGF, JSON)', required=True)
@click.pass_context
def embed_ms_file(ctx, model, ref_spectra, project_accession, input_file):
  """
  Encoding the spectra into a 32-vector file
  :param ctx: Context environment from click
  :param model: Model trained to improve the similarity search
  :param ref_spectra: reference spectra from spectra cluster.
  :param project_accession: ProteomeXchange accession for the file
  :param input_file: project file to be index.
  :return:
  """
  encode_and_embed_spectra(model, ref_spectra, project_accession, input_file)


@click.command('make-index',
               short_help="Commandline to make faiss indexIVFFLat index for every MS/MS spectrum\'s 32 features vector")
@click.option('--database_ids_usi_file', '-d', help='Input database ids file which is named database_ids_usi.csv',
              required=True)
@click.option('--embedded_spectra_path', '-e', type=click.Path(exists=True),
              help='Path of embedded spectra file, the files end with "-embedded.txt" would be used to create index file',
              required=True)
@click.option('--output', '-o', help='Output index file', required=True)
@click.pass_context
def make_index(ctx, database_ids_usi_file, embedded_spectra_path, output):
  """
  Make index in faiss from the 32-vector file.
  :param ctx: Context environment from click
  :param database_ids_usi_file: database with ids and usi files
  :param embedded_spectra_path: embedded spectra 32-vector file.
  :param output: index file.
  """
  index_maker = FaissWriteIndex()
  index_maker.create_index_for_embedded_spectra(database_ids_usi_file, embedded_spectra_path, output)


@click.command('merge-indexes', short_help="Commandline to merge multiple index files")
@click.argument('input_indexes', nargs=-1)
@click.argument('output', nargs=1)
@click.pass_context
def merge_indexes(ctx, input_indexes, output):
  """
  Merge input indexes into a big database.
  :param ctx: Context environment from click
  :param input_indexes: input indexes file
  :param output: output database
  :return:
  """
  index_maker = FaissWriteIndex()
  index_maker.merge_indexes(input_indexes, output=output)


@click.command('range-search', short_help="Commandline to range search query embedded spectra against index file")
@click.option('--index_file', '-i', help='Index file', required=True)
@click.option('--index_ids_usi_file', '-u', help="Index's ids_usi data file", required=True)
@click.option('--embedded_spectra', '-e', help='Input embedded spectra file', required=True)
@click.option('--lower_threshold', '-lt', help='Lower radius for range search', default=0.0)
@click.option('--upper_threshold', '-ut', help='Upper radius for range search', default=0.07)
@click.option('--nprobe', '-n', help='Faiss index nprobe', default=DEFAULT_IVF_NLIST)
@click.option('--output', '-o', help='Output file of range search result', required=True)
@click.pass_context
def range_search(ctx, index_file, index_ids_usi_file, embedded_spectra, lower_threshold, upper_threshold, nprobe, output):
  """
  Search into database different spectra file.
  :param ctx: Context environment from click
  :param index_file: Input database
  :param embedded_spectra: embedded spectra to be search
  :param threshold: threshold to be use for search, default 0.1
  :param output: out file including all the usi that have been found.
  :return:
  """
  index_searcher = FaissIndexSearch()
  index_searcher.execute_range_search(index_file, index_ids_usi_file, embedded_spectra, lower_threshold, upper_threshold, nprobe, output)



@click.command('auto-range-search', short_help="Commandline to automatically range search query embedded spectra against index file")
@click.option('--model', '-m', help='Input embedder model file',
              default="./dleamse_model_references/080802_20_1000_NM500R_model.pkl")
@click.option('--ref_spectra', '-r', help='Input 500 reference spectra file',
              default="./dleamse_model_references/0722_500_rf_spectra.mgf")
@click.option('--project_accession', '-p', help='ProteomeXchange dataset accession', default="Project_ID")
@click.option('--index_file', '-i', help='Index file',  default="./dleamse_model_references/database.index")
@click.option('--index_ids_usi_file', '-u', help="Index's ids_usi data file",  default="./dleamse_model_references/database_ids_usi.csv")
@click.option('--query_spectra', '-e', help='Input MS File (supported: mzML, MGF, JSON)', required=True)
@click.option('--lower_threshold', '-lt', help='Lower radius for range search', default=0.0)
@click.option('--upper_threshold', '-ut', help='Upper radius for range search', default=0.07)
@click.option('--nprobe', '-n', help='Faiss index nprobe', default=DEFAULT_IVF_NLIST)
@click.option('--output', '-o', help='Output file of range search result', required=True)
@click.pass_context
def auto_range_search(ctx, model, ref_spectra, index_file, project_accession, index_ids_usi_file, query_spectra, lower_threshold, upper_threshold, nprobe, output):
  """
  Automatically search for different spectrum files in the database.
  python mslookup.py auto-range-search -i ./dleamse_model_references/database.index -u ./dleamse_model_references/database_ids_usi.csv -e testdata/query.json -lt 0.0 -ut 0.099 -o testdata/minor.json
  python mslookup.py auto-range-search -i ./testdata/db.index -u ./testdata/db_ids_usi.csv -e testdata/query.json -lt 0.0 -ut 0.099 -o testdata/minor.json
  :param ctx: Context environment from click
  :param model: Model trained to improve the similarity search
  :param ref_spectra: reference spectra from spectra cluster.
  :param project_accession: ProteomeXchange accession for the file
  :param index_file: Input database
  :param query_spectra: raw spectra file to be search
  :param threshold: threshold to be use for search, default 0.1
  :param output: out file including all the usi that have been found.
  :return:
  """

  #embedded spectra
  encode_and_embed_spectra(model, ref_spectra, project_accession, query_spectra)
  dirname, filename = os.path.split(os.path.abspath(query_spectra))
  if query_spectra.endswith(".mgf"):
    embedded_spectra = dirname + "/" + str(filename.strip(".mgf")) + "_embedded.txt"
  elif query_spectra.endswith(".mzML"):
    embedded_spectra = dirname + "/" + str(filename.strip(".mzML")) + "_embedded.txt"
  else:
    embedded_spectra = dirname + "/" + str(filename.strip(".json")) + "_embedded.txt"
  #search
  index_searcher = FaissIndexSearch()
  index_searcher.execute_range_search(index_file, index_ids_usi_file, embedded_spectra, lower_threshold, upper_threshold, nprobe, output)


@click.command('onestop-range-search', short_help="Commandline to one-stop range search query embedded spectra against index file")
@click.option('--model', '-m', help='Input embedder model file',
              default="./dleamse_model_references/080802_20_1000_NM500R_model.pkl")
@click.option('--ref_spectra', '-r', help='Input 500 reference spectra file',
              default="./dleamse_model_references/0722_500_rf_spectra.mgf")
@click.option('--project_accession', '-p', help='ProteomeXchange dataset accession', default="Project_ID")

@click.option('--database_ids_usi_file', '-d', help='Input database ids file which is named database_ids_usi.csv',
              required=True)
@click.option('--outputdb', '-odb', help='Output index file',required=True)

@click.option('--query_spectra', '-e', help='Input MS File (supported: mzML, MGF, JSON)', required=True)
@click.option('--library_spectra', '-ls', help='Input MS File (supported: mzML, MGF, JSON) to create library', required=True)
@click.option('--lower_threshold', '-lt', help='Lower radius for range search', default=0.0)
@click.option('--upper_threshold', '-ut', help='Upper radius for range search', default=0.07)
@click.option('--nprobe', '-n', help='Faiss index nprobe', default=DEFAULT_IVF_NLIST)
@click.option('--output', '-o', help='Output file of range search result', required=True)
@click.pass_context
def onestop_range_search(ctx, model, ref_spectra, project_accession, database_ids_usi_file, outputdb, query_spectra, library_spectra, lower_threshold, upper_threshold, nprobe, output):
  """
  Automatically search for different spectrum files in the database.
  python mslookup.py onestop-range-search -d testdata/db.csv -odb testdata/db.index -ls testdata/PXD015890_114263_ArchiveSpectrum.json -e testdata/query.json -lt 0.0 -ut 0.099 -o testdata/minor.json
  python mslookup.py onestop-range-search -d /home/qinchunyuan/528xiyangMakeForTest/test0825/db.csv -odb /home/qinchunyuan/528xiyangMakeForTest/test0825/db.index -ls /home/qinchunyuan/528xiyangMakeForTest/test0825/PXD015890_114263_ArchiveSpectrum.json -e /home/qinchunyuan/528xiyangMakeForTest/test0825/query.json -lt 0.0 -ut 0.099 -o /home/qinchunyuan/528xiyangMakeForTest/test0825/minor.json
  :param ctx: Context environment from click
  :param model: Model trained to improve the similarity search
  :param ref_spectra: reference spectra from spectra cluster.
  :param project_accession: ProteomeXchange accession for the file
  :param database_ids_usi_file: database with ids and usi files
  #:param embedded_spectra_path: embedded spectra 32-vector file.
  :param outputdb: index file.
  #:param index_file: Input database
  :param query_spectra: raw spectra file to be search
  :param threshold: threshold to be use for search, default 0.1
  :param output: out file including all the usi that have been found.
  :return:
  """

  dirname, filename = os.path.split(os.path.abspath(query_spectra))
  if query_spectra.endswith(".mgf"):
    embedded_spectra = dirname + "/" + str(filename.strip(".mgf")) + "_embedded.txt"
    embedded_spectra_path = dirname + "/"
  elif query_spectra.endswith(".mzML"):
    embedded_spectra = dirname + "/" + str(filename.strip(".mzML")) + "_embedded.txt"
    embedded_spectra_path = dirname + "/"
  else:
    embedded_spectra = dirname + "/" + str(filename.strip(".json")) + "_embedded.txt"
    embedded_spectra_path = dirname + "/"

  if "../" in outputdb:
  	index_ids_usi_file = ".."+outputdb.strip('.index') + '_ids_usi.csv'
  else:
  	index_ids_usi_file = outputdb.strip('index').strip(".") + '_ids_usi.csv'
    
  #create library-embedded
  encode_and_embed_spectra(model, ref_spectra, project_accession, library_spectra)
  # make index
  index_maker = FaissWriteIndex()
  index_maker.create_index_for_embedded_spectra(database_ids_usi_file, embedded_spectra_path, outputdb)


  #embedded spectra query
  encode_and_embed_spectra(model, ref_spectra, project_accession, query_spectra)
  
  #search
  index_searcher = FaissIndexSearch()
  index_searcher.execute_range_search(outputdb, index_ids_usi_file, embedded_spectra, lower_threshold, upper_threshold, nprobe, output)


cli.add_command(embed_ms_file)
cli.add_command(make_index)
cli.add_command(merge_indexes)
cli.add_command(range_search)
cli.add_command(auto_range_search)
cli.add_command(onestop_range_search)

if __name__ == "__main__":
  cli()
