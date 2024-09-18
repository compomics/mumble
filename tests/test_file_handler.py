from io import StringIO
from unittest.mock import MagicMock, mock_open, patch
import numpy as np
import pandas as pd
from psm_utils import Peptidoform
from rustyms import RawSpectrum, RawPeak
import pytest
from mumble.file_handler import _SpectrumFileHandler, _MetadataParser

class TestSpectrumFileHandler:

    @pytest.fixture
    def setup_spectrum_handler(self):
        """Fixture to set up a mock _SpectrumFileHandler object."""
        spectrum_handler = MagicMock(spec=_SpectrumFileHandler)
        return spectrum_handler
    
    @pytest.fixture
    def setup_rawspectrum(self):
        rawSpectrum = RawSpectrum(
            title="Some spectra",
            num_scans=3,
            rt=1.0,
            precursor_charge=10,
            precursor_mass=500.5,
            mz_array=[100.0, 200.0, 300.0],
            intensity_array=[10.0, 20.0, 30.0],
        )
        return rawSpectrum

    def test_init_with_mgf_file(self):
        """Test initialization with a valid MGF file."""
        with patch.object(_SpectrumFileHandler, '_parse_mgf') as mock_parse_mgf:
            handler = _SpectrumFileHandler("test.mgf")
            mock_parse_mgf.assert_called_once()
            assert handler.file_type == "MGF"
    
    def test_init_with_mzml_file(self):
        """Test initialization with a valid mzML file."""
        with patch.object(_SpectrumFileHandler, '_parse_mzml') as mock_parse_mzml:
            handler = _SpectrumFileHandler("test.mzml")
            mock_parse_mzml.assert_called_once()
            assert handler.file_type == "MZML"
    
    def test_init_with_unsupported_file_type(self):
        """Test initialization with an unsupported file type raises a ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format. Only MGF and mzML are supported."):
            _SpectrumFileHandler("test.txt")

    def test_parse_mgf(self):
        """Test parsing an MGF file and storing spectra."""
        mock_mgf_data = [
            {
                "m/z array": [100.0, 200.0, 300.0],
                "intensity array": [10.0, 20.0, 30.0],
                "params": {"title": "spec1", "pepmass": [500.5]}
            },
            {
                "m/z array": [110.0, 220.0],
                "intensity array": [15.0, 25.0],
                "params": {"title": "spec2", "pepmass": [600.6]}
            }
        ]
        
        expected_spectra = [[(100,10),(200,20),(300,30)],
                            [(110,15),(220,25)]]   
        
        # Create a mock MGF object
        mock_mgf = MagicMock()
        mock_mgf.__enter__.return_value = iter(mock_mgf_data)
        mock_mgf.__exit__.return_value = False
        
        with patch("pyteomics.mgf.MGF", return_value=mock_mgf):
            handler = _SpectrumFileHandler("test.mgf")

            assert len(handler.spectra) == 2
            assert isinstance(handler.spectra["spec1"], RawSpectrum)
            assert isinstance(handler.spectra["spec2"], RawSpectrum)

            assert handler.spectra["spec1"].mass == 500.5
            assert handler.spectra["spec1"].num_scans == 3
            
            assert handler.spectra["spec2"].mass == 600.6
            assert handler.spectra["spec2"].num_scans == 2

            # check rawPeacks in spectrum/mz_array for each spectrum
            for spectrumCounter, rawSpectrum in enumerate(handler.spectra.values()):            
                for peakCounter, rawPeak in enumerate(rawSpectrum.spectrum):
                    assert rawPeak.mz == expected_spectra[spectrumCounter][peakCounter][0], f"Mismatch at spectrum {spectrumCounter}, peak {peakCounter}: expected mz {expected_spectra[spectrumCounter][peakCounter][0]}, got {rawPeak.mz}"
                    assert rawPeak.intensity == expected_spectra[spectrumCounter][peakCounter][1], f"Mismatch at spectrum {spectrumCounter}, peak {peakCounter}: expected intensity {expected_spectra[spectrumCounter][peakCounter][1]}, got {rawPeak.intensity}"
    
    def test_parse_mzml(self):
        """Test parsing an mzML file and storing spectra."""
        mock_mzml_data = [
            {
                "id": "spec1",
                "m/z array": [100.0, 200.0, 300.0],
                "intensity array": [10.0, 20.0, 30.0],
                "precursorList": {
                    "precursor": [{
                        "selectedIonList": {"selectedIon": [{"selected ion m/z": 500.5}]}
                    }]
                }
            },
            {
                "id": "spec2",
                "m/z array": [110.0, 220.0],
                "intensity array": [15.0, 25.0],
                "precursorList": {
                    "precursor": [{
                        "selectedIonList": {"selectedIon": [{"selected ion m/z": 600.6}]}
                    }]
                }
            }
        ]
        
        expected_spectra = [[(100,10),(200,20),(300,30)],
                            [(110,15),(220,25)]]   

        # Create a mock MzML object
        mock_mzml = MagicMock()
        mock_mzml.__enter__.return_value = iter(mock_mzml_data)
        mock_mzml.__exit__.return_value = False
        
        with patch("pyteomics.mzml.MzML", return_value=mock_mzml):
            handler = _SpectrumFileHandler("test.mzml")


            assert len(handler.spectra) == 2
            assert isinstance(handler.spectra["spec1"], RawSpectrum)
            assert isinstance(handler.spectra["spec2"], RawSpectrum)

            assert handler.spectra["spec1"].mass == 500.5
            assert handler.spectra["spec1"].num_scans == 3
            
            
            assert handler.spectra["spec2"].mass == 600.6
            assert handler.spectra["spec2"].num_scans == 2

            # check rawPeacks in spectrum/mz_array for each spectrum
            for spectrumCounter, rawSpectrum in enumerate(handler.spectra.values()):            
                for peakCounter, rawPeak in enumerate(rawSpectrum.spectrum):
                    assert rawPeak.mz == expected_spectra[spectrumCounter][peakCounter][0], f"Mismatch at spectrum {spectrumCounter}, peak {peakCounter}: expected mz {expected_spectra[spectrumCounter][peakCounter][0]}, got {rawPeak.mz}"
                    assert rawPeak.intensity == expected_spectra[spectrumCounter][peakCounter][1], f"Mismatch at spectrum {spectrumCounter}, peak {peakCounter}: expected intensity {expected_spectra[spectrumCounter][peakCounter][1]}, got {rawPeak.intensity}"


    def test_get_spectrum_found(self, setup_rawspectrum):
        """Test retrieving a spectrum by its ID (spectrum found)."""
        mock_spectrum = setup_rawspectrum

        handler = _SpectrumFileHandler("test.mgf")
        handler.spectra = {"spec1": mock_spectrum}

        spectrum = handler.get_spectrum("spec1")
        
        expected_spectrum = [(100,10),(200,20),(300,30)]
        
        assert spectrum is mock_spectrum
    
        # check data based on mock_spectrum
        assert spectrum.mass == 500.5
        assert spectrum.charge == 10.0
        assert spectrum.num_scans == 3
        assert spectrum.rt == 1.0
        
        # check spectrum/mz_array
        for i, rawPeak in enumerate(spectrum.spectrum):
            assert rawPeak.mz == expected_spectrum[i][0]
            assert rawPeak.intensity == expected_spectrum[i][1]
        
    def test_get_spectrum_not_found(self, setup_rawspectrum):
        """Test retrieving a spectrum by its ID (spectrum not found)."""
        handler = _SpectrumFileHandler("test.mzml")
        mock_spectrum = setup_rawspectrum
        handler.spectra = {"spec1": mock_spectrum}

        spectrum = handler.get_spectrum("unknown_spec_id")
        assert spectrum is None


    
class TestMetadataParser:
    
    @pytest.fixture
    def setup_metadata_handler(self):

        metadata_handler = _MetadataParser()
        return metadata_handler
    
    def test_parse_csv_file_valid(self, setup_metadata_handler):

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