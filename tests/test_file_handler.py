from io import StringIO
from unittest.mock import MagicMock, mock_open, patch
import pandas as pd
from psm_utils import Peptidoform
import pytest

from mumble.file_handler import _MetadataParser

    
class TestMetadataParser:
    
    @pytest.fixture
    def setup_metadata_handler(self):

        metadata_handler = _MetadataParser()
        return metadata_handler
    
    def test_parse_csv_file_valid(self, setup_metadata_handler):

        # psm_handler, mod_handler, psm = setup_psmhandler
        metadata_handler = setup_metadata_handler

        # Mock CSV data
        csv_data = """peptidoform\tspectrum_id\tprecursor_mz
        ART[Deoxy]HR/2\tspec1\t214.1
        ABCD/2\tspec2\t300.2
        """

        with patch("builtins.open", mock_open(read_data=csv_data)), \
            patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_data), delimiter="\t")):
            peptidoforms = metadata_handler.parse_csv_file("dummy_file.tsv")

        assert len(peptidoforms) == 2
        assert peptidoforms[0].peptidoform == Peptidoform("ART[Deoxy]HR/2")
        assert peptidoforms[0].spectrum_id == "spec1"
        assert peptidoforms[0].precursor_mz == 214.1

    def test_parse_csv_file_missing_columns(self, setup_metadata_handler):
        metadata_handler = setup_metadata_handler

        # Mock CSV data with missing 'precursor_mz' column
        csv_data = """peptidoform\tspectrum_id
        ART[Deoxy]HR\tspec1
        ABCD\tspec2
        """

        with patch("builtins.open", mock_open(read_data=csv_data)), \
             patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_data), delimiter="\t")):
            peptidoforms = metadata_handler.parse_csv_file("dummy_file.tsv", delimiter="\t")

        assert peptidoforms == []  # Should return an empty list due to missing columns

    def test_parse_csv_file_file_not_found(self, setup_metadata_handler):
        metadata_handler = setup_metadata_handler

        with patch("builtins.open", side_effect=FileNotFoundError):
            peptidoforms = metadata_handler.parse_csv_file("non_existent_file.tsv", delimiter="\t")

        assert peptidoforms == []  # Should return an empty list due to FileNotFoundError

    def test_parse_csv_file_empty_file(self, setup_metadata_handler):
        metadata_handler = setup_metadata_handler

        # Mock empty CSV data
        csv_data = """peptidoform\tspectrum_id\tprecursor_mz"""

        with patch("builtins.open", mock_open(read_data=csv_data)), \
            patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_data), delimiter="\t")):
            peptidoforms = metadata_handler.parse_csv_file("dummy_file.tsv", delimiter="\t")

        assert peptidoforms == []  # Should return an empty list due to empty file