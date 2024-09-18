import logging
import pandas as pd

from psm_utils import PSM


class _FileHandler:
    """
    Class to handle spectrum files and retrieve spectra for PSM annotations.
    Supports MGF and mzML file formats.
    """
      
    def parse_csv_file(self, file_name: str, delimiter: str = "\t") -> list:
        """
        Write simple input that takes tsv or csv file with: peptidoform, spectrum_id, precursor_mz and write to Peptidoforms
        
        Args:
        file_name (str): Path to the CSV or TSV file.
        delimiter (str, optional): Delimiter used in the file. Defaults to "\t"

        return:
            list of Peptidoforms
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
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        peptidoforms = [
            PSM(peptidoform=row["peptidoform"], spectrum_id=row["spectrum_id"], precursor_mz=row["precursor_mz"])
            for _, row in df.iterrows()
        ]

        return peptidoforms