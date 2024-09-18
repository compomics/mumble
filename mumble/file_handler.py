import logging
import pandas as pd
from psm_utils import PSM
from pyteomics import mgf, mzml
from rustyms import spectrum as rusty_spectrum


class SpectrumFileHandler:
    """
    Class to handle spectrum files (MGF or mzML) and retrieve spectra by spectrum ID.
    """
    
    def __init__(self, spectrum_file: str):
        self.spectrum_file = spectrum_file
        self.spectra = None
        self.file_type = None

        if spectrum_file.endswith(".mgf"):
            self.file_type = "MGF"
            self.spectra = mgf.IndexedMGF(spectrum_file)
        elif spectrum_file.endswith(".mzml"):
            self.file_type = "MZML"
            self.spectra = mzml.IndexedMzML(spectrum_file)
        else:
            raise ValueError("Unsupported file format. Only MGF and mzML are supported.")
    
    def get_spectrum(self, spectrum_id: str):
        """
        Retrieve the spectrum by its ID from the spectrum file.
        
        Args:
            spectrum_id (str): The ID of the spectrum.
        
        Returns:
            rusty_spectrum.Spectrum: Retrieved spectrum or None if not found.
        """
        try:
            if self.file_type == "MGF":
                spectrum = self.spectra[spectrum_id]
            elif self.file_type == "MZML":
                spectrum = self.spectra.get_by_id(spectrum_id)
            else:
                logging.error("No spectra file loaded.")
                return None

            return rusty_spectrum.Spectrum(
                mz_values=spectrum["m/z array"], 
                intensities=spectrum["intensity array"], 
                precursor_mz=spectrum.get("precursorList", {}).get("precursor", [{}])[0]
                               .get("selectedIonList", {}).get("selectedIon", [{}])[0]
                               .get("selected ion m/z", None)
            )
        except KeyError:
            logging.error(f"Spectrum ID {spectrum_id} not found in the file.")
            return None


class MetadataParser:
    """
    Class to parse metadata files (CSV/TSV) containing PSM information.
    """
    
    @staticmethod
    def parse_psm_file(file_name: str, delimiter: str = "\t") -> list:
        """
        Parse a CSV or TSV file containing peptidoform, spectrum_id, and precursor_mz.
        
        Args:
            file_name (str): Path to the CSV or TSV file.
            delimiter (str, optional): Delimiter used in the file. Defaults to "\t".

        Returns:
            list of PSMs: A list of PSM objects with the necessary information.
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
            PSM(peptidoform=row["peptidoform"], spectrum_id=row["spectrum_id"], precursor_mz=row["precursor_mz"])
            for _, row in df.iterrows()
        ]

        return peptidoforms
