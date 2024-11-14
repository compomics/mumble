import click

from mumble import PSMHandler

import time


@click.command("cli", context_settings={"show_default": True})
@click.argument(
    "input_file",
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
    "--aa-combinations",
    "-a",
    help="Number of amino acid combinations to add as modification REQUIRES fasta_file",
    type=click.INT,
    default=0,
    show_default=True,
)
@click.option(
    "--fasta-file",
    "-f",
    help="Path to a fasta file",
    type=click.Path(exists=True),
    default=None,
)
@click.option(
    "--mass-error",
    "-m",
    help="Mass error for the mass shift",
    type=click.FLOAT,
    default=0.02,
    show_default=True,
)
@click.option(
    "--output-file",
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
    "--generate-modified-decoys",
    help="Parse modifications for decoys in modified PSMlist",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--keep-original",
    help="Keep the original PSMs in the modified PSMlist",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "--num-modifications-combination",
    help="Maximum number of modifications per combination. All lower numbers will be included as well.",
    type=click.INT,
    default=1,
    show_default=True,
)
@click.option(
    "--exclude-mutations",
    help="If set, modifications with the classification 'AA substitution' will be excluded.",
    type=click.BOOL,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--config-file",
    help="Path to a config file",
    type=click.Path(exists=True),
    default=None,
)
def main(
    input_file,
    filetype_read,
    aa_combinations,
    fasta_file,
    mass_error,
    output_file,
    filetype_write,
    generate_modified_decoys,
    keep_original,
    num_modifications_combination,
    exclude_mutations,
    config_file,
):
    """
    Finding the perfect match for your mass shift.\n
    __________

    The `main` function is the entry point for processing Peptide Spectrum Matches (PSMs) with potential mass modifications. It reads the input PSM file, identifies possible modifications based on mass shifts found by the search engine, and generates new PSM entries with these modifications. The function can handle different file types, apply amino acid combination modifications, and incorporate decoy sequences if specified. The resulting modified PSM list can be output in various formats, allowing for easy integration into downstream analysis pipelines.
    """
    t1 = time.perf_counter()
    psm_handler = PSMHandler(
        aa_combinations=aa_combinations,
        fasta_file=fasta_file,
        mass_error=mass_error,
        combination_length=num_modifications_combination,
        exclude_mutations=exclude_mutations,
    )
    modified_psm_list = psm_handler.add_modified_psms(
        input_file,
        psm_file_type=filetype_read,
        generate_modified_decoys=generate_modified_decoys,
        keep_original=keep_original,
        config_file=config_file,
    )
    t2 = time.perf_counter()
    runtime_ms = (t2 - t1) * 1000
    print(f"Computed in {runtime_ms:0.2f} milliseconds\n")

    psm_handler.write_modified_psm_list(
        modified_psm_list, output_file=output_file, psm_file_type=filetype_write
    )


if __name__ == "__main__":
    main()
