import logging
import os

import click

from dleamse.dleamse_encode_and_embed import encode_and_embed_spectra
from dleamse.dleamse_encode_and_embed import SiameseNetwork2
from dleamse.dleamse_faiss_index_writer import FaissWriteIndex

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(context_settings=CONTEXT_SETTINGS)
def cli():
    """This is the main tool that give access to all commands and options provided by the mslookup and dleamse algorithms"""


class AppConfigException(object):
    def __init__(self, value):
        super(AppConfigException, self).__init__(value)


@click.command('encode-ms-file',
               short_help='Commandline to encode every MS spectrum in a file into a 32 features vector')
@click.option('--model',              '-m', help='Input embedder model file',
                                      default="./dleamse_model_references/080802_20_1000_NM500R_model.pkl")
@click.option('--project_accession',  '-p', help='ProteomeXchange dataset accession', default="Project_ID")
@click.option('--input',              '-i', help='Input MS File (supported: mzML, MGF, JSON)', required=True)
@click.option('--ref_spectra',        '-r', help='Input 500 reference spectra file', default="./dleamse_model_references/0722_500_rf_spectra.mgf")
@click.option('--output',             '-o', help='Output vectors file, its default path is the same as input file', default="outputfile.csv")
@click.option('--miss_record',        '-s', help='Bool, record charge missed spectra', default="True")
@click.option('--use_gpu',            '-g', help='Bool, use gpu or not', default="False")
@click.option('--make_faiss_index',   '-f', help='Make faiss index', default="False")
@click.pass_context
def encode_ms_file(ctx, model: str, project_accession: str, input: str, ref_spectra: str,
                   output: str, miss_record: str,
                   use_gpu: str, make_faiss_index: str):
    model = model
    prj = project_accession
    input_file = input
    ref_spectra = ref_spectra
    miss_record = miss_record

    dirname, filename = os.path.split(os.path.abspath(input_file))
    output = output
    output_file, miss_record_file, index_ids_file, index_file = None, None, None, None

    if output:
        dirname, filename = os.path.split(os.path.abspath(input_file))
        if filename.endswith(".mgf"):
            output_file = dirname + os.path.sep + filename.strip(".mgf") + "_embedded.npy"
        elif filename.endswith(".mzML"):
            output_file = dirname + os.path.sep + filename.strip(".mzML") + "_embedded.npy"
        else:
            output_file = dirname + os.path.sep + filename.strip(".json") + "_embedded.npy"

        if miss_record:
            if filename.endswith(".mgf"):
                miss_record_file = dirname + os.path.sep + filename.strip(".mgf") + "_miss_record.txt"
            elif filename.endswith(".mzML"):
                miss_record_file = dirname + os.path.sep + filename.strip(".mzML") + "_miss_record.txt"
            else:
                miss_record_file = dirname + os.path.sep + filename.strip(".json") + "_miss_record.txt"

        if make_faiss_index:
            if filename.endswith(".mgf"):
                index_file = dirname + os.path.sep + filename.strip(".mgf") + ".index"
            elif filename.endswith(".mzML"):
                index_file = dirname + os.path.sep + filename.strip(".mzML") + ".index"
            else:
                index_file = dirname + os.path.sep + filename.strip(".json") + ".index"

    else:
        output_file = output
        dirname, filename = os.path.split(os.path.abspath(output))
        if miss_record:
            if filename.endswith(".mgf"):
                miss_record_file = dirname + os.path.sep + filename.strip(".mgf") + "_miss_record.txt"
            elif filename.endswith(".mzML"):
                miss_record_file = dirname + os.path.sep + filename.strip(".mzML") + "_miss_record.txt"
            else:
                miss_record_file = dirname + os.path.sep + filename.strip(".json") + "_miss_record.txt"

        if make_faiss_index:
            if filename.endswith(".mgf"):
                index_ids_file = dirname + os.path.sep + filename.strip(".mgf") + "_ids.txt"
                index_file = dirname + os.path.sep + filename.strip(".mgf") + ".index"
            elif filename.endswith(".mzML"):
                index_ids_file = dirname + os.path.sep + filename.strip(".mzML") + "_ids.txt"
                index_file = dirname + os.path.sep + filename.strip(".mzML") + ".index"
            else:
                index_ids_file = dirname + os.path.sep + filename.strip(".json") + "_ids.txt"
                index_file = dirname + os.path.sep + filename.strip(".json") + ".index"

    embedded_spectra_data = encode_and_embed_spectra(model, prj, input_file, ref_spectra)

    if make_faiss_index:
        index_maker = FaissWriteIndex(embedded_spectra_data, index_ids_file, index_file)


cli.add_command(encode_ms_file)

if __name__ == "__main__":
    cli()
