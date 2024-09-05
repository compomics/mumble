import pytest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
from io import StringIO
from collections import namedtuple
from psm_utils import PSMList, PSM, Peptidoform
from pyteomics import proforma
from pyteomics.fasta import IndexedFASTA

from mumble.mumble import _ModificationHandler, PSMHandler


class TestPSMHandler:

    @pytest.fixture
    def setup_psmhandler(self):
        # Fixture for setting up PSMHandler with mocked dependencies
        psm = PSM(
            peptidoform="ART[Deoxy]HR",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
        )

        mod_handler = MagicMock(spec=_ModificationHandler)
        psm_handler = PSMHandler(aa_combinations=0, fasta_file=None, mass_error=0.02)
        psm_handler.modification_handler = mod_handler

        return psm_handler, mod_handler, psm

    def test_find_mod_locations(self, setup_psmhandler):
        psm_handler, _, _ = setup_psmhandler

        peptidoform = MagicMock()
        peptidoform.properties = {"n_term": "some_mod", "c_term": None}
        peptidoform.parsed_sequence = [
            ("A", None),
            ("C", "mod1"),
            ("D", None),
            ("E", None),
            ("F", "mod2"),
        ]

        locations = psm_handler._find_mod_locations(peptidoform)
        assert locations == ["N-term", 1, 4]

    def test_return_mass_shifted_peptidoform(self, setup_psmhandler):
        psm_handler, mod_handler, psm = setup_psmhandler

        mod_handler.aa_sub_dict = {"His->Ala": ("H", "A")}

        new_peptidoform_1 = psm_handler._return_mass_shifted_peptidoform(
            ("C-term", "Ahx2+Hsl"), psm.peptidoform
        )
        new_peptidoform_2 = psm_handler._return_mass_shifted_peptidoform(
            (3, "His->Ala"), psm.peptidoform
        )

        assert new_peptidoform_1 is not None
        assert new_peptidoform_1.properties["c_term"] == [proforma.process_tag_tokens("Ahx2+Hsl")]
        assert new_peptidoform_2 is not None
        assert new_peptidoform_2 == Peptidoform("ART[Deoxy]AR")

    def test_create_new_psm(self, setup_psmhandler):
        psm_handler, _, psm = setup_psmhandler

        new_peptidoform = "ARTAR"
        new_psm = psm_handler._create_new_psm(psm, new_peptidoform)

        assert new_psm is not None
        assert new_psm.peptidoform == new_peptidoform

    def test_get_modified_peptidoforms(self, setup_psmhandler):
        psm_handler, mod_handler, psm = setup_psmhandler

        mod_handler.aa_sub_dict = {"His->Ala": ("H", "A")}

        mod_handler.localize_mass_shift.return_value = [("N-term", "Acetyl")]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep_original=True)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 2
        assert new_psms[0].peptidoform.properties["n_term"] == ["Acetyl"]
        assert new_psms[1] == psm

        mod_handler.localize_mass_shift.return_value = [(1, "Carbamyl"), (4, "Carbamyl")]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep_original=False)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 2
        assert new_psms[0].peptidoform == Peptidoform("AR[Carbamyl]T[Deoxy]HR")
        assert new_psms[1].peptidoform == Peptidoform("ART[Deoxy]HR[Carbamyl]")

    def test_add_modified_psms(self, setup_psmhandler):
        psm_handler, mod_handler, psm = setup_psmhandler

        psm_list = [psm]
        mod_handler.localize_mass_shift.return_value = [("N-term", "mod1")]
        new_psm_list = psm_handler.add_modified_psms(psm_list)

        assert isinstance(new_psm_list, PSMList)
        assert len(new_psm_list) > 1

    def test_parse_csv_file_valid(self, setup_psmhandler):

        # psm_handler, mod_handler, psm = setup_psmhandler
        psm_handler = setup_psmhandler[0]

        # Mock CSV data
        csv_data = """peptidoform\tspectrum_id\tprecursor_mz
        ART[Deoxy]HR/2\tspec1\t214.1
        ABCD/2\tspec2\t300.2
        """

        with patch("builtins.open", mock_open(read_data=csv_data)), \
            patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_data), delimiter="\t")):
            peptidoforms = psm_handler.parse_csv_file("dummy_file.tsv")

        assert len(peptidoforms) == 2
        assert peptidoforms[0].peptidoform == Peptidoform("ART[Deoxy]HR/2")
        assert peptidoforms[0].spectrum_id == "spec1"
        assert peptidoforms[0].precursor_mz == 214.1

    def test_parse_csv_file_missing_columns(self, setup_psmhandler):
        psm_handler = setup_psmhandler[0]

        # Mock CSV data with missing 'precursor_mz' column
        csv_data = """peptidoform\tspectrum_id
        ART[Deoxy]HR\tspec1
        ABCD\tspec2
        """

        with patch("builtins.open", mock_open(read_data=csv_data)), \
             patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_data), delimiter="\t")):
            peptidoforms = psm_handler.parse_csv_file("dummy_file.tsv", delimiter="\t")

        assert peptidoforms == []  # Should return an empty list due to missing columns

    def test_parse_csv_file_file_not_found(self, setup_psmhandler):
        psm_handler = setup_psmhandler[0]

        with patch("builtins.open", side_effect=FileNotFoundError):
            peptidoforms = psm_handler.parse_csv_file("non_existent_file.tsv", delimiter="\t")

        assert peptidoforms == []  # Should return an empty list due to FileNotFoundError

    def test_parse_csv_file_empty_file(self, setup_psmhandler):
        psm_handler = setup_psmhandler[0]

        # Mock empty CSV data
        csv_data = """peptidoform\tspectrum_id\tprecursor_mz"""

        with patch("builtins.open", mock_open(read_data=csv_data)), \
            patch("pandas.read_csv", return_value=pd.read_csv(StringIO(csv_data), delimiter="\t")):
            peptidoforms = psm_handler.parse_csv_file("dummy_file.tsv", delimiter="\t")

        assert peptidoforms == []  # Should return an empty list due to empty file


class TestModificationHandler:

    @pytest.fixture
    def setup_modhandler(self):
        # Fixture for setting up _ModificationHandler
        psm = PSM(
            peptidoform="ART[Deoxy]HR/3",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
            precursor_mz=228.4614,
        )
        mod_handler = _ModificationHandler(mass_error=0.02)
        return mod_handler, psm

    def test_get_unimod_database(self, setup_modhandler):
        mod_handler, _ = setup_modhandler
        mod_handler.get_unimod_database()

        assert mod_handler.modification_df is not None

    def test_add_amino_acid_combinations(self, setup_modhandler):
        mod_handler, _ = setup_modhandler

        mod_handler._add_amino_acid_combinations(2)
        assert mod_handler.modification_df is not None
        assert "YP" in mod_handler.modification_df["name"].values
        assert "Q" in mod_handler.modification_df["name"].values
        assert (
            mod_handler.modification_df[mod_handler.modification_df["name"] == "YP"][
                "rounded_mass"
            ].values[0]
            == 260
        )
        assert (
            mod_handler.modification_df[mod_handler.modification_df["name"] == "YP"][
                "monoisotopic_mass"
            ].values[0]
            == 260.116093
        )

    def test_get_localisation(self, setup_modhandler):
        mod_handler, _ = setup_modhandler

        psm = PSM(
            peptidoform="QART[Deoxy]HRQ/3",
            spectrum_id="some_spectrum",
        )

        # Example input data
        modification_name = "mod1"
        residue_list = ["R", "N-term", "C-term", "Q", "protein_level"]
        restrictions = ["anywhere", "N-term", "C-term", "N-term", "anywhere"]

        # Mock the check_protein_level method
        mod_handler.check_protein_level = MagicMock(return_value=[("prepeptide", "mod1")])

        # Expected output
        Localised_mass_shift = namedtuple("Localised_mass_shift", ["loc", "modification"])
        expected_output = [
            Localised_mass_shift(2, "mod1"),
            Localised_mass_shift(5, "mod1"),  # R in the sequence
            Localised_mass_shift("N-term", "mod1"),  # N-term modification
            Localised_mass_shift("C-term", "mod1"),  # C-term modification
            Localised_mass_shift("N-term", "mod1"),  # Q in the sequence
            Localised_mass_shift("prepeptide", "mod1"),  # protein level modification
        ]

        # Call the method
        result = mod_handler.get_localisation(psm, modification_name, residue_list, restrictions)

        # Assertions
        assert result == expected_output

        psm = PSM(
            peptidoform="KTIEVFDPDADTW/2",
            spectrum_id="some_spectrum",
        )

        # Example input data
        modification_name = "mod1"
        residue_list = ["F"]
        restrictions = ["N-term"]

        # Expected output
        expected_output = []

        # Call the method
        result = mod_handler.get_localisation(psm, modification_name, residue_list, restrictions)

        # Assertions
        assert result == expected_output

    # TODO: refactor after changes in localize_mass_shift
    def test_localize_mass_shift_combination_length_1(self,setup_modhandler):
        # Create an instance of the handler
        mod_handler, _ = setup_modhandler

        # Mock necessary attributes related to the UniMod database
        # Mock necessary attributes on the instance
        mod_handler.name_to_mass_residue_dict = {
            "Carbamyl": MagicMock(residues=['C'], restrictions=None),
            "Acetyl": MagicMock(residues=['N-term'], restrictions=None),
        }
        mod_handler.monoisotopic_masses = [43.005814, 42.010565]
        mod_handler.modifications_names = [("Carbamyl",), ("Acetyl",)]
        # mod_handler._binary_range_search = MagicMock(return_value=(0, 2))
        mod_handler.get_localisation = MagicMock(side_effect=[
            [{"loc": 1, "modification": "Carbamyl"}],  # First call for Carbamyl
            [{"loc": "N-term", "modification": "Acetyl"}],  # Second call for Acetyl
        ])

        # Mock PSM object with necessary attributes
        psm = PSM(
            peptidoform="ART[Deoxy]HR/3",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
            precursor_mz=208.79446854107334,
        )
        original_precursor_mz = psm.precursor_mz

        # Test case 1: Carbamyl
        psm.precursor_mz = original_precursor_mz + (43.005814 / 3)
        localized_modifications = mod_handler.localize_mass_shift(psm)

        assert localized_modifications is not None
        assert localized_modifications[0].Localised_mass_shift[0].loc == 1
        assert localized_modifications[0].Localised_mass_shift[0].modification == "Carbamyl"

        # Test case 2: Acetyl
        # psm.precursor_mz = original_precursor_mz + (42.010565 / 3)
        # localized_modifications = mod_handler.localize_mass_shift(psm)

        # assert localized_modifications is not None
        # assert localized_modifications[0].Localised_mass_shift[0].loc == "N-term"
        # assert localized_modifications[0].Localised_mass_shift[0].modification == "Acetyl"

    # def test_localize_mass_shift2(self, setup_modhandler):
    #     mod_handler, _ = setup_modhandler

    #     psm = PSM(
    #         peptidoform="ART[Deoxy]HR/3",
    #         spectrum_id="some_spectrum",
    #         is_decoy=False,
    #         protein_list=["some_protein"],
    #         precursor_mz=208.79446854107334,
    #     )
    #     orginal_precursor_mz = psm.precursor_mz

    #     psm.precursor_mz = orginal_precursor_mz + (((43.005814 + 42.010565) / 3))
    #     localized_modifications = mod_handler.localize_mass_shift(psm)
    #     print(localized_modifications)

    def test_check_protein_level(self, setup_modhandler):
        mod_handler, psm = setup_modhandler

        mod_handler.fasta_file = MagicMock(IndexedFASTA)
        mod_handler.fasta_file.__getitem__.return_value.sequence = "RASSLCTPARTHRQVMHUW"

        additional_aa = "TP"
        results = mod_handler.check_protein_level(psm, additional_aa)
        assert ("prepeptide", "TP") in results

        additional_aa = "Q"
        results = mod_handler.check_protein_level(psm, additional_aa)
        assert ("postpeptide", "Q") in results


    @pytest.fixture
    def setup_modhandler_with_data(self):
        # Setup a _ModificationHandler instance with a sample modification DataFrame
        mod_handler = _ModificationHandler(mass_error=0.02)
        data = {
            "name": ["mod1", "mod2", "mod3"],
            "monoisotopic_mass": [10.0, 40.0, 20.0],
        }
        mod_handler.modification_df = pd.DataFrame(data)
        return mod_handler
    
    def test_generate_modifications_combinations_lists_length_1(self, setup_modhandler_with_data):
        mod_handler = setup_modhandler_with_data

        # Test the function with combination_length=1
        masses, combinations = mod_handler._generate_modifications_combinations_lists(combination_length=1)

        # Expected results
        expected_masses = [
            10.0,
            20.0,
            40.0,
        ]
        expected_combinations = [
            ("mod1",),
            ("mod3",),
            ("mod2",),
        ]

        # Assertions
        assert masses == expected_masses
        assert combinations == expected_combinations
    
    def test_generate_modifications_combinations_lists_length_2(self, setup_modhandler_with_data):
        mod_handler = setup_modhandler_with_data

        # Test the function with combination_length=2
        masses, combinations = mod_handler._generate_modifications_combinations_lists(combination_length=2)

        # Expected results
        expected_masses = [
            10.0,
            20.0,
            10.0 + 20.0,
            40.0,
            10.0 + 40.0,
            20.0 + 40.0,
        ]
        expected_combinations = [
            ("mod1",),
            ("mod3",),
            ("mod1","mod3"),
            ("mod2",),
            ("mod1", "mod2"),
            ("mod3", "mod2"),
        ]

        # Assertions for masses
        assert masses == expected_masses

        # Convert tuples to frozensets for comparison
        expected_combinations_set = set(frozenset(x) for x in expected_combinations)
        combinations_set = set(frozenset(x) for x in combinations)

        # Assertions for combinations (ignoring order within tuples and order of tuples in the list)
        assert combinations_set == expected_combinations_set

    def test_generate_modifications_combinations_lists_length_3(self, setup_modhandler_with_data):
        mod_handler = setup_modhandler_with_data

        # Test the function with combination_length=3
        masses, combinations = mod_handler._generate_modifications_combinations_lists(combination_length=3)

        # Expected results
        expected_masses = [
            10.0,
            20.0,
            10.0 + 20.0,
            40.0,
            10.0 + 40.0,
            20.0 + 40.0,
            10.0 + 20.0 + 40.0
        ]
        expected_combinations = [
            ("mod1",),
            ("mod3",),
            ("mod1","mod3"),
            ("mod2",),
            ("mod1", "mod2"),
            ("mod3", "mod2"),
            ("mod1", "mod2", "mod3"),
        ]

        # Assertions for masses
        assert masses == expected_masses

        # Convert tuples to frozensets for comparison
        expected_combinations_set = set(frozenset(x) for x in expected_combinations)
        combinations_set = set(frozenset(x) for x in combinations)

        # Assertions for combinations (ignoring order within tuples and order of tuples in the list)
        assert combinations_set == expected_combinations_set


    def test_generate_modifications_combinations_lists_empty(self, setup_modhandler_with_data):
        mod_handler = setup_modhandler_with_data

        # Test the function with an empty DataFrame
        mod_handler.modification_df = pd.DataFrame(columns=["name", "monoisotopic_mass"])

        masses, combinations = mod_handler._generate_modifications_combinations_lists(combination_length=2)

        # Assertions
        assert masses == []
        assert combinations == []


if __name__ == "__main__":
    pytest.main()
