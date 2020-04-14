import logging
import os

import click

from dleamse.dleamse_encode_and_embed import encode_and_embed_spectra
from dleamse.dleamse_encode_and_embed import SiameseNetwork2
from dleamse.dleamse_faiss_index_writer import FaissWriteIndex
from dleamse.dleamse_faiss_index_search import FaissIndexSearch

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
  """This is the main tool that give access to all commands and options provided by the mslookup and dleamse algorithms"""


class AppConfigException(object):
  def __init__(self, value):
    super(AppConfigException, self).__init__(value)


@click.command('encode-ms-file',
               short_help='Commandline to encode and embed every MS/MS spectrum in a file into a 32 features vector')
@click.option('--model', '-m', help='Input embedder model file',
              default="./dleamse/dleamse_model_references/080802_20_1000_NM500R_model.pkl")
@click.option('--ref_spectra', '-r', help='Input 500 reference spectra file',
              default="./dleamse/dleamse_model_references/0722_500_rf_spectra.mgf")
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
@click.option('--database_ids_file', '-d', help='Input database ids file which is named database_ids.npy',
              required=True)
@click.option('--embedded_spectra_path', '-e', type=click.Path(exists=True),
              help='Path of embedded spectra file, the files end with "-embedded.txt" would be used to create index file',
              required=True)
@click.option('--output', '-o', help='Output index file', required=True)
@click.pass_context
def make_index(ctx, database_ids_file, embedded_spectra_path, output):
  """
  Make index in faiss from the 32-vector file.
  :param ctx: Context environment from click
  :param database_ids_file: database with ids and usi files
  :param embedded_spectra_path: embedded spectra 32-vector file.
  :param output: index file.
  """
  index_maker = FaissWriteIndex()
  index_maker.create_index_for_embedded_spectra(database_ids_file, embedded_spectra_path, output)


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
@click.option('--embedded_spectra', '-e', help='Input embedded spectra file', required=True)
@click.option('--threshold', '-t', help='Radius for range search', default=0.1)
@click.option('--output', '-o', help='Output file of range search result', required=True)
@click.pass_context
def range_search(ctx, index_file, embedded_spectra, threshold, output):
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
  index_searcher.execute_range_search(index_file, embedded_spectra, threshold, output)


cli.add_command(embed_ms_file)
cli.add_command(make_index)
cli.add_command(merge_indexes)
cli.add_command(range_search)

if __name__ == "__main__":
  cli()
