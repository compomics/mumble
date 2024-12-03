import logging
import numpy as np
import pandas as pd
from psm_utils import PSM, PSMList
from pyteomics import mgf, mzml
from rustyms import RawSpectrum


class _SpectrumFileHandler:
    """
    A class to handle spectrum files (MGF or mzML) and retrieve spectra by spectrum ID.

    This class parses spectrum files, stores the spectra, and provides methods to retrieve
    spectra based on PSM (Peptide-Spectrum Match) information.

    Attributes:
        spectrum_file (str): Path to the spectrum file (MGF or mzML).
        spectra (dict): A dictionary storing parsed spectra, keyed by spectrum ID.
        file_type (str): The type of the spectrum file ('MGF' or 'MZML').

    Methods:
        get_spectrum_from_psm(psm): Retrieve a spectrum for a given PSM.
        get_spectra_from_psm_list(psmList): Retrieve spectra for a list of PSMs.
        get_all_spectra(): Retrieve all parsed spectra.

    Raises:
        ValueError: If an unsupported file format is provided.

    Note:
        This class uses the pyteomics library for parsing MGF and mzML files.
        Parsed spectra are stored as `rustyms.RawSpectrum` objects.
    """

    def __init__(self, spectrum_file: str):
        self.spectrum_file = spectrum_file
        self.spectra = {}  # Initialize an empty dictionary to hold the spectra
        self.file_type = None

        # Determine file type and call the appropriate parser
        if spectrum_file.lower().endswith(".mgf"):
            self.file_type = "MGF"
            self._parse_mgf()
        elif spectrum_file.lower().endswith(".mzml"):
            self.file_type = "MZML"
            self._parse_mzml()
        else:
            raise ValueError("Unsupported file format. Only MGF and mzML are supported.")

    def _parse_mgf(self):
        """
        Parse an MGF (Mascot Generic Format) file and store each spectrum as a RawSpectrum object.

        This method reads the MGF file specified in self.spectrum_file, extracts relevant information
        for each spectrum (including spectrum ID, precursor mass, retention time, and charge),
        and stores each spectrum as a RawSpectrum object in the self.spectra dictionary.

        Raises:
            Exception: If there's an error during the parsing process, it's logged as an error.
        """
        try:
            with mgf.MGF(self.spectrum_file) as spectra:
                for spectrum in spectra:
                    spectrum_id = spectrum["params"].get(
                        "title", "Unknown"
                    )  # Extract spectrum ID from the MGF params
                    precursor_mass = spectrum["params"].get("pepmass", [None])[
                        0
                    ]  # Extract precursor mass

                    # Extract retention time
                    rt = 0.0
                    if "rtinseconds" in spectrum["params"]:
                        rt = float(spectrum["params"]["rtinseconds"])
                    elif "retention time" in spectrum["params"]:
                        rt = float(spectrum["params"]["retention time"])

                    # Extract precursor charge
                    precursor_charge = 0
                    if "charge" in spectrum["params"]:
                        charge_str = spectrum["params"]["charge"]
                        precursor_charge = int(
                            charge_str.strip("+")
                        )  # Remove '+' and convert to int

                    # Create a RawSpectrum object using required fields and additional attributes
                    self.spectra[spectrum_id] = RawSpectrum(
                        title=spectrum_id,
                        num_scans=len(spectrum["m/z array"]),
                        rt=rt,
                        precursor_charge=precursor_charge,
                        mz_array=np.array(spectrum["m/z array"]),
                        intensity_array=np.array(spectrum["intensity array"]),
                        precursor_mass=precursor_mass,
                    )
            logging.info(f"Parsed {len(self.spectra)} spectra from {self.spectrum_file}")
        except Exception as e:
            logging.error(f"Error parsing MGF file {self.spectrum_file}: {e}")

    def _parse_mzml(self):
        """
        Parse an mzML file and store each spectrum as a RawSpectrum object.

        This method reads the mzML file specified in self.spectrum_file, extracts relevant information
        for each spectrum (including spectrum ID, precursor mass, retention time, and charge),
        and stores each spectrum as a RawSpectrum object in the self.spectra dictionary.

        Raises:
            Exception: If there's an error during the parsing process, it's logged as an error.
        """
        try:
            with mzml.MzML(self.spectrum_file) as spectra:
                for spectrum in spectra:
                    spectrum_id = spectrum.get(
                        "id", None
                    )  # Get the spectrum ID from the mzML spectrum
                    precursor_mass = 0.0
                    precursor_charge = 0
                    rt = 0.0

                    # Extract precursor mass and charge if available
                    if "precursorList" in spectrum and spectrum["precursorList"]:
                        precursor = spectrum["precursorList"]["precursor"][0]
                        if "selectedIonList" in precursor:
                            selected_ion = precursor["selectedIonList"]["selectedIon"][0]
                            precursor_mass = selected_ion.get("selected ion m/z", 0.0)
                            precursor_charge = int(selected_ion.get("charge state", 0))

                    # Extract retention time
                    if "scanList" in spectrum and spectrum["scanList"]:
                        scan = spectrum["scanList"]["scan"][0]
                        for cv_param in scan.get("cvParam", []):
                            if (
                                cv_param.get("accession") == "MS:1000016"
                            ):  # accession for scan start time
                                rt = float(cv_param.get("value", 0.0))
                                break

                    # Create a RawSpectrum object using required fields and additional attributes
                    self.spectra[spectrum_id] = RawSpectrum(
                        title=spectrum_id,
                        num_scans=len(spectrum["m/z array"]),
                        rt=rt,
                        precursor_charge=precursor_charge,
                        mz_array=np.array(spectrum["m/z array"]),
                        intensity_array=np.array(spectrum["intensity array"]),
                        precursor_mass=precursor_mass,
                    )
            logging.info(f"Parsed {len(self.spectra)} spectra from {self.spectrum_file}")
        except Exception as e:
            logging.error(f"Error parsing mzML file {self.spectrum_file}: {e}")

    def get_spectrum_from_psm(self, psm: PSM):
        """
        Retrieve a RawSpectrum for a PSM by its ID.

        Args:
            psm (PSM): psm object

        Returns:
            RawSpectrum: The retrieved spectrum or None if not found.
        """
        return self.spectra.get(psm.spectrum_id, None)

    def get_spectra_from_psm_list(self, psmList: PSMList):
        """
        Retrieve all spectra for a PSMList.

        Args:
            psmList (PSMList): A list of PSM objects.

        Returns:
            list: A list of RawSpectrum objects corresponding to the PSMs.
                None is included for any spectra not found.
        """
        return [self.spectra.get(psm.spectrum_id) for psm in psmList]

    def get_all_spectra(self):
        """
        Retrieve all parsed spectra.

        Returns:
            dict: A dictionary of all parsed spectra, where keys are spectrum IDs
                and values are RawSpectrum objects.
        """
        return self.spectra


class _MetadataParser:
    """
    Class to parse metadata files (CSV/TSV) containing PSM information.
    """

    @staticmethod
    def parse_csv_file(file_name: str, delimiter: str = "\t") -> list:
        """
        Parse a CSV or TSV file containing PSM information and create PSM objects.

        Args:
            file_name (str): Path to the CSV or TSV file.
            delimiter (str, optional): Delimiter used in the file. Defaults to "\t".

        Returns:
            list: A list of PSM objects containing information from the file.
                Each PSM object includes peptidoform, spectrum_id, and precursor_mz.

        Raises:
            FileNotFoundError: If the specified file is not found.
            pd.errors.EmptyDataError: If the file is empty.
            pd.errors.ParserError: If there's an error parsing the file.

        Notes:
            The file must contain at least the following columns:
            'peptidoform', 'spectrum_id', and 'precursor_mz'.
            If any of these columns are missing, an error is logged and an empty list is returned.
        """

        try:
            df = pd.read_csv(file_name, delimiter=delimiter)
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            return []
        except pd.errors.EmptyDataError as e:
            logging.error(f"Empty data: {e}")
            return []
        except pd.errors.ParserError as e:
            logging.error(f"Parsing error: {e}")
            return []

        required_columns = {"peptidoform", "spectrum_id", "precursor_mz"}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            logging.error(f"Missing required columns: {missing}")
            return []

        # Clean up any whitespace in DataFrame
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Create a list of PSM objects from the DataFrame rows
        peptidoforms = [
            PSM(
                peptidoform=row["peptidoform"],
                spectrum_id=row["spectrum_id"],
                precursor_mz=row["precursor_mz"],
            )
            for _, row in df.iterrows()
        ]

        return peptidoforms
