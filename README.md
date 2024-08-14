# Mumble

**Finding the perfect unimod match for your mass shifted PSM**
## Overview

The PSM Modification Handler is a Python-based tool designed to find candidate unimod modifications for mass shifts. The tool allows users to apply modifications to PSMs, localize mass shifts, and generate lists of modified PSMs.

## Features

- **PSM Modification**: Apply specific modifications to PSMs and generate modified PSM lists.
- **Mass Shift Localization**: Identify potential modifications in peptides by localizing mass shifts.
- **Flexible Input/Output**: Read PSMs from various file formats, modify them, and write the results to different output formats.
- **Customizable Modifications**: Supports the addition of amino acid combinations and handles custom modifications through the Unimod database.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Required Libraries

Install the required Python library using the following command:

```bash
pip install mumble
```


### Basic Usage

Here's a quick example of how to use the PSM Modification Handler for single PSMs:

```python
>>> from mumble import PSMHandler
>>> from psm_utils import PSM

>>> # Initialize the PSMHandler
>>> psm_handler = PSMHandler(aa_combinations=0, fasta_file=None, mass_error=0.02)

>>> # Create a minimal PSM to generate modified version from
>>> psm = PSM(
...     peptidoform="ARTHR/3",
...     precursor_mz=228.129628 # Required information
... )
>>> # Generate proteoforms for given PSM with a certain MZ
>>> modified_proteoforms = PSMHandler.get_modified_peptidoforms_list(psm, keep_original=False)


>>> # Write the modified PSM list to a file
>>> psm_handler.write_modified_psm_list(modified_proteoforms, output_file="modified_proteoforms.tsv", psm_file_type="tsv")

>>> print(modified_proteoforms)
# [
#     PSM(
#         peptidoform="[Acetyl]-ARTHR/3"
#         precursor_mz=228.129628
#     )
# ]
```
Here's a quick example of how to use the PSM Modification Handler for PSM lists:
```python
>>> # Or load a PSM list (from a file or PSMList object)
>>> psm_list = psm_handler.parse_psm_list("path/to/psm_file.mzid", psm_file_type="mzid")

>>> # Add modified PSMs to the list
>>> modified_psm_list = psm_handler.add_modified_psms(psm_list, generate_modified_decoys=False, keep_original=True)

>>> # Write the modified PSM list to a file
>>> psm_handler.write_modified_psm_list(modified_psm_list, output_file="modified_psms.tsv", psm_file_type="tsv")
```
For more information on PSM objects and PSM lists visit [psm_utils](https://github.com/compomics/psm_utils)
## Testing

The project includes unit tests using `pytest` to ensure code reliability.

### Running Tests

To run the tests, simply use the following command:

```bash
pytest
```

### Contributing


Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PSMUtils**: For providing core utilities for PSM handling. ([psm_utils](https://github.com/compomics/psm_utils))
- **Pyteomics**: For offering tools to handle mass spectrometry data. ([pyteomics](https://github.com/levitsky/pyteomics))

