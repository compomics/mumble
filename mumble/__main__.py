import click

from mumble import PSMHandler


@click.command("cli", context_settings={"show_default": True})
@click.option(
    "--input_file",
    "-i",
    help="Path to the input file",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--filetype-read",
    "-fr",
    help="Type of the input file to read with PSM_utlis.io.read_file",
    type=click.STRING,
    default="infer",
)
@click.option(
    "--aa_combinations",
    "-a",
    help="Number of amino acid combinations to add as modification",
    type=click.INT,
    default=0,
    show_default=True,
)
@click.option(
    "--fasta_file",
    "-f",
    help="Path to the fasta file",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--mass_error",
    "-m",
    help="Mass error for the mass shift",
    type=click.FLOAT,
    default=0.02,
    show_default=True,
)
@click.option(
    "--output_file",
    "-o",
    help="Path to the output file",
    type=click.STRING,
    default=None,
)
@click.option(
    "--filetype-write",
    "-fw",
    help="Type of the output file to write with PSM_utlis.io.write_file",
    type=click.STRING,
    default="tsv",
    show_default=True,
)
@click.option(
    "--add_decoys",
    help="Parse modifications for decoys in modified PSMlist",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--keep",
    help="Keep the original PSMs in the modified PSMlist",
    is_flag=True,
    default=False,
    show_default=True,
)
def main(
    input_file,
    filetype_read,
    aa_combinations,
    fasta_file,
    mass_error,
    output_file,
    filetype_write,
    add_decoys,
    keep,
):
    """
    Where your mass shift can find its unimod match.
    __________

    The `main` function is the entry point for processing Peptide Spectrum Matches (PSMs) with potential mass modifications. It reads the input PSM file, identifies possible modifications based on mass shifts found by the search engine, and generates new PSM entries with these modifications. The function can handle different file types, apply amino acid combination modifications, and incorporate decoy sequences if specified. The resulting modified PSM list can be output in various formats, allowing for easy integration into downstream analysis pipelines.
    """
    psm_handler = PSMHandler(
        aa_combinations=aa_combinations, fasta_file=fasta_file, mass_error=mass_error
    )
    modified_psm_list = psm_handler.add_modified_psms(
        input_file, psm_file_type=filetype_read, add_decoys=add_decoys, keep=keep
    )
    psm_handler.write_modified_psm_list(
        modified_psm_list, output_file=output_file, psm_file_type=filetype_write
    )


if __name__ == "__main__":
    main()