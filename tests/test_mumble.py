import os

import pytest
from unittest.mock import MagicMock
import pandas as pd
from collections import namedtuple
from psm_utils import PSMList, PSM, Peptidoform
from psm_utils.utils import mz_to_mass
from pyteomics import proforma
from pyteomics.mass import calculate_mass
from pyteomics.fasta import IndexedFASTA

from mumble.mumble import _ModificationHandler, PSMHandler

# Define named tuples globally
Localised_mass_shift = namedtuple("Localised_mass_shift", ["loc", "modification"])
Modification_candidate = namedtuple("Modification_candidate", ["Localised_mass_shifts"])


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

    @pytest.fixture
    def setup_psm(self):
        psm = PSM(
            peptidoform="VTFTETPENGSKW/2",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
            precursor_mz="748.8581250320699",
        )

        return psm

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

        new_peptidoform_1_list = psm_handler._return_mass_shifted_peptidoform(
            [
                Modification_candidate(
                    Localised_mass_shifts=[
                        Localised_mass_shift(loc="C-term", modification="Ahx2+Hsl")
                    ]
                )
            ],
            psm.peptidoform,
        )
        new_peptidoform_2_list = psm_handler._return_mass_shifted_peptidoform(
            [
                Modification_candidate(
                    Localised_mass_shifts=[Localised_mass_shift(loc=3, modification="His->Ala")]
                )
            ],
            psm.peptidoform,
        )

        assert new_peptidoform_1_list is not None
        assert new_peptidoform_1_list[0].properties["c_term"] == [
            proforma.process_tag_tokens("Ahx2+Hsl")
        ]
        assert new_peptidoform_2_list is not None
        assert new_peptidoform_2_list[0] == Peptidoform("ART[Deoxy]AR")

    def test_create_new_psm(self, setup_psmhandler):
        psm_handler, _, psm = setup_psmhandler

        new_peptidoform = "ARTAR"
        new_psm = psm_handler._create_new_psm(psm, new_peptidoform)

        assert new_psm is not None
        assert new_psm.peptidoform == new_peptidoform

    def test_get_modified_peptidoforms(self, setup_psmhandler):
        psm_handler, mod_handler, psm = setup_psmhandler

        mod_handler.aa_sub_dict = {"His->Ala": ("H", "A")}

        # test 2 seprate Modification_candidate
        mod_handler.localize_mass_shift.return_value = [
            Modification_candidate(
                Localised_mass_shifts=[Localised_mass_shift("N-term", "Acetyl")]
            )
        ]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep_original=True)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 2
        assert new_psms[1].peptidoform.properties["n_term"] == ["Acetyl"]
        assert new_psms[0] == psm

        # test 2 seprate Modification_candidate
        mod_handler.localize_mass_shift.return_value = [
            Modification_candidate(Localised_mass_shifts=[Localised_mass_shift(1, "Carbamyl")]),
            Modification_candidate(Localised_mass_shifts=[Localised_mass_shift(4, "Carbamyl")]),
        ]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep_original=False)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 2
        assert new_psms[0].peptidoform == Peptidoform("AR[Carbamyl]T[Deoxy]HR")
        assert new_psms[1].peptidoform == Peptidoform("ART[Deoxy]HR[Carbamyl]")

        # test combination of Localised_mass_shift inside 1 Modification_candidate
        mod_handler.localize_mass_shift.return_value = [
            Modification_candidate(
                Localised_mass_shifts=[
                    Localised_mass_shift(1, "Carbamyl"),
                    Localised_mass_shift(4, "Carbamyl"),
                ]
            )
        ]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep_original=False)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 1  # 1 combined psm expected
        assert new_psms[0].peptidoform == Peptidoform(
            "AR[Carbamyl]T[Deoxy]HR[Carbamyl]"
        )  # combination of mods

    def test_add_modified_psms(self, setup_psmhandler):
        psm_handler, mod_handler, psm = setup_psmhandler

        psm_list = [psm]
        mod_handler.localize_mass_shift.return_value = [
            Modification_candidate(Localised_mass_shifts=[Localised_mass_shift("N-term", "mod1")])
        ]
        new_psm_list = psm_handler.add_modified_psms(psm_list, keep_original=True)

        assert isinstance(new_psm_list, PSMList)
        assert len(new_psm_list) > 1

    def test_tool_combination_length_1(self, setup_psm):

        psm_handler = PSMHandler(combination_length=1, exclude_mutations=False)

        # retrigger get_unimod_database
        cache_file = psm_handler.modification_handler.cache._get_cache_file_path()
        psm_handler.modification_handler.cache._load_or_generate_data(
            cache_file, force_reload=True
        )
        psm = setup_psm

        result_psm_list = psm_handler.get_modified_peptidoforms_list(psm)

        # order of PSMs/peptidoforms doesn't matter
        expected_Peptidoforms = frozenset(
            [Peptidoform("VTFTETPEDGSKW/2"), Peptidoform("VTFTETPEN[Deamidated]GSKW/2")]
        )
        result_Peptidoforms = frozenset(psm.peptidoform for psm in result_psm_list)

        # original psm should not be returned by default
        assert psm not in result_psm_list

        assert result_Peptidoforms == expected_Peptidoforms

    def test_tool_combination_length_2(self, setup_psm):

        psm_handler = PSMHandler(combination_length=2)

        # retrigger get_unimod_database
        cache_file = psm_handler.modification_handler.cache._get_cache_file_path()
        psm_handler.modification_handler.cache._load_or_generate_data(
            cache_file, force_reload=True
        )

        psm = setup_psm
        result_psm_list = psm_handler.get_modified_peptidoforms_list(psm)
        result_peptidoforms = [result_psm.peptidoform for result_psm in result_psm_list]

        # all expected peptidoforms with 1 mod
        expected_single_mod_Peptidoforms = [
            Peptidoform("VTFTETPEDGSKW/2"),
            Peptidoform("VTFTETPEN[Deamidated]GSKW/2"),
        ]

        # some of the expected peptidoforms with 2 mods
        expected_double_mod_Peptidoforms = [
            Peptidoform("VTFTETPESGSRW/2"),
            Peptidoform("VTFTETPE[Oxidation]NGSIW/2"),
            Peptidoform("VTFTNTPENGSK[Oxidation]W/2"),
            Peptidoform("VTFTETP[Pro->HAVA]EPGSKW/2"),
            Peptidoform("VTFT[Oxidation]ETPEVGSKW/2"),
            Peptidoform("VT[Methyl]FTETPETGSKW/2"),
        ]

        # check every peptidoform is unique, PSM non hashable so set(result_psm_list) results in error
        assert len(result_peptidoforms) == len(set(result_peptidoforms))

        assert len(result_psm_list) == 186

        assert all(
            peptidoform in result_peptidoforms for peptidoform in expected_single_mod_Peptidoforms
        )
        assert all(
            peptidoform in result_peptidoforms for peptidoform in expected_double_mod_Peptidoforms
        )

    def test_tool_keep_original(self, setup_psm):
        # psm_handler = setup_psmhandler[0]
        psm_handler = PSMHandler(combination_length=1)

        psm = setup_psm

        result_psm_list = psm_handler.get_modified_peptidoforms_list(psm, keep_original=True)

        assert psm in result_psm_list
        assert len(result_psm_list) == 3


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

    @pytest.fixture
    def setup_modhandler_with_data(self):
        # Setup a _ModificationHandler instance with a sample modification DataFrame
        mod_handler = _ModificationHandler(mass_error=0.02)
        data = {
            "name": ["mod1", "mod2", "mod3"],
            "monoisotopic_mass": [10.0, 40.0, 20.0],
        }
        mod_handler.cache.modification_df = pd.DataFrame(data)

        return mod_handler

    @pytest.fixture
    def get_cache_paths(self):
        cache_dir = "modification_cache"
        cache_file_path = os.path.join(
            cache_dir, "length_1_exclude_mutations_False.pkl"
        )  # just for testing
        return cache_dir, cache_file_path

    def test_get_unimod_database(self, setup_modhandler):
        mod_handler, _ = setup_modhandler
        # mod_handler.ModificationCache.get_unimod_database()

        assert mod_handler.modification_df is not None

    def test_add_amino_acid_combinations(self, setup_modhandler):
        mod_handler, _ = setup_modhandler

        mod_handler._add_amino_acid_combinations(2)
        assert mod_handler.modification_df is not None
        assert "YP" in mod_handler.modification_df["name"].values
        assert "Q" in mod_handler.modification_df["name"].values
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
        mod_handler.check_protein_level = MagicMock(return_value=[("pepeptide", "mod1")])

        # Expected output
        expected_output = {
            Localised_mass_shift(2, "mod1"),
            Localised_mass_shift(5, "mod1"),  # R in the sequence
            Localised_mass_shift("N-term", "mod1"),  # N-term modification
            Localised_mass_shift("C-term", "mod1"),  # C-term modification
            Localised_mass_shift("N-term", "mod1"),  # Q in the sequence
            Localised_mass_shift("pepeptide", "mod1"),  # protein level modification
        }

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
        expected_output = set()

        # Call the method
        result = mod_handler.get_localisation(psm, modification_name, residue_list, restrictions)

        # Assertions
        assert result == expected_output

    # Helper function to compare expected and actual localisations
    def localisation_matches(self, expected, actual):
        """
        This function checks if the expected localisation is found within the actual Modification_candidate.
        It ignores the order of the Localised_mass_shift items.
        """
        expected_set = {frozenset(localisation.items()) for localisation in expected}
        actual_set = {
            frozenset({"loc": loc.loc, "modification": loc.modification}.items())
            for loc in actual.Localised_mass_shifts
        }

        return expected_set == actual_set

    # TODO: refactor after changes in localize_mass_shift
    def test_localize_mass_shift_combination_length_1(self, setup_modhandler):
        # Create an instance of the handler
        mod_handler, _ = setup_modhandler

        # Mock necessary attributes related to the UniMod database
        mod_handler.name_to_mass_residue_dict = {
            "mod1": MagicMock(residues=["M"], restrictions=None),
            "mod2": MagicMock(residues=["C"], restrictions=None),
            "mod3": MagicMock(residues=["N-term"], restrictions=None),
            "mod4": MagicMock(residues=["S", "T", "Y"], restrictions=None),
        }

        # Setting up 4 modifications: 2 close to the target mass shift, and 2 far
        mod_handler.monoisotopic_masses = [
            15.994915,  # Far from target
            43.005814,  # Close to target
            43.015814,  # Close to target
            79.966331,  # Far from target
        ]

        mod_handler.modifications_names = [
            ("mod1",),  # Far
            ("mod2",),  # Close§
            ("mod3",),  # Close
            ("mod4",),  # Far
        ]

        # Mock get_localisation to return valid positions for the modifications
        mod_handler.get_localisation = MagicMock(
            side_effect=[
                [
                    Localised_mass_shift(loc=1, modification="mod2"),
                    Localised_mass_shift(loc="N-term", modification="mod2"),
                ],  # Carbamyl localised at position 1 or N-terminal
                [
                    Localised_mass_shift(loc="N-term", modification="mod3"),
                    Localised_mass_shift(loc=1, modification="mod3"),
                ],  # Acetyl at N-terminal or 1
                # These won't be called since they are far from the target
                [Localised_mass_shift(loc=3, modification="Oxidation")],
                [Localised_mass_shift(loc=5, modification="Phospho")],
            ]
        )

        # Mock PSM object with necessary attributes
        psm = PSM(
            peptidoform="ART[Deoxy]HR/3",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
            precursor_mz=208.79446854107334,  # Mock value
        )

        # Test Case: Localize mass shift for Carbamyl/Acetyl modifications (close masses)
        original_precursor_mz = psm.precursor_mz
        target_mass_shift = 43.005814  # We will aim for a mass shift close to this

        # Set the precursor m/z so it corresponds to a mass shift close to 43.005814 (Carbamyl)
        psm.precursor_mz = original_precursor_mz + (target_mass_shift / 3)

        localized_modifications = mod_handler.localize_mass_shift(psm)

        # Assertions to ensure only close modifications are tested
        assert localized_modifications is not None
        assert len(localized_modifications) == 4

        expected_localisations = [
            [
                {"loc": 1, "modification": "mod2"},
            ],
            [
                {"loc": "N-term", "modification": "mod3"},
            ],
            [
                {"loc": "N-term", "modification": "mod2"},
            ],
            [{"loc": 1, "modification": "mod3"}],
        ]

        # Check if every expected_localisations is present
        for expected in expected_localisations:
            match_found = any(
                self.localisation_matches(expected, actual) for actual in localized_modifications
            )
            assert (
                match_found
            ), f"Expected localisation {expected} not found in localized_modifications."

        # Ensure far masses are not tested (Oxidation and Phospho should be skipped)
        assert mod_handler.get_localisation.call_count == 2  # Only close ones should be called

    def test_localize_mass_shift_combination_length_2(self, setup_modhandler):
        # Create an instance of the handler
        mod_handler, _ = setup_modhandler

        # Value's not important for test this since get_localisation gets mocked. Needed so function doesn't crash when fetching the residues before calling get_localisation.
        mod_handler.name_to_mass_residue_dict = {
            "mod3": MagicMock(residues=["C"], restrictions=None),
            "mod2": MagicMock(residues=["N-term"], restrictions=None),
            "mod1": MagicMock(residues=["M"], restrictions=None),
            "mod4": MagicMock(residues=["S", "T", "Y"], restrictions=None),
            "mod5": MagicMock(residues=["S", "T", "Y"], restrictions=None),
        }

        # If this is updated the order of get_localisation MagicMock might need to be changed too
        modifications = [
            ("mod1", 15.994915, "Far", None, "M"),
            ("mod2", 43.005814, "Close_combined", None, "K"),
            ("mod3", 43.015814, "Close_combined", None, "K"),
            ("mod4", 79.966331, "Far", None, "S"),
            ("mod5", (43.005814 + 43.005814), "Close_alone", None, "K"),
        ]

        # Create the DataFrame
        mod_handler.cache.modification_df = pd.DataFrame(
            modifications,
            columns=[
                "name",
                "monoisotopic_mass",
                "classification",
                "restriction",
                "residue",
            ],
        )

        mod_handler.monoisotopic_masses, mod_handler.modifications_names = (
            mod_handler.cache._generate_modifications_combinations_lists(2)
        )

        # Mock get_localisation to return valid positions for the modifications
        # These should be in order with the combinations sorted on weights so if the list modifications changes this will probably need to be updated too
        mod_handler.get_localisation = MagicMock(
            side_effect=[
                [
                    Localised_mass_shift(loc=1, modification="mod2"),
                    Localised_mass_shift(loc="N-term", modification="mod2"),
                ],  # Carbamyl localised at position 1 or N-terminal
                [Localised_mass_shift(loc=1, modification="mod5")],
                [
                    Localised_mass_shift(loc="N-term", modification="mod3"),
                    Localised_mass_shift(loc=1, modification="mod3"),
                ],  # Acetyl at N-terminal or 1
                # These won't be called since they are far from the target
                [Localised_mass_shift(loc=3, modification="mod1")],
                [Localised_mass_shift(loc=5, modification="mod4")],
            ]
        )

        # Mock PSM object with necessary attributes
        psm = PSM(
            peptidoform="ART[Deoxy]HR/3",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
            precursor_mz=208.79446854107334,  # Mock value
        )

        # Test Case: Localize mass shift for Carbamyl/Acetyl modifications (close masses)
        original_precursor_mz = psm.precursor_mz
        target_mass_shift = 43.005814 + 43.005814  # We will aim for a mass shift close to this

        # Set the precursor m/z so it corresponds to a mass shift close to 43.005814 (Carbamyl)
        psm.precursor_mz = original_precursor_mz + (target_mass_shift / 3)

        localized_modifications = mod_handler.localize_mass_shift(psm)

        # Assertions to ensure only close modifications are tested
        assert localized_modifications is not None
        assert (
            len(localized_modifications) == 5
        )  # len((mod2,mod2);(mod2,mod2);(mod2,mod3);(mod3,mod2);(mod5,))

        expected_localisations = [
            [{"loc": 1, "modification": "mod2"}, {"loc": "N-term", "modification": "mod2"}],
            [{"loc": "N-term", "modification": "mod2"}, {"loc": 1, "modification": "mod3"}],
            [{"loc": "N-term", "modification": "mod3"}, {"loc": 1, "modification": "mod2"}],
            [{"loc": 1, "modification": "mod5"}],
        ]

        # Check if every expected_localisations is present
        for expected in expected_localisations:
            match_found = any(
                self.localisation_matches(expected, actual) for actual in localized_modifications
            )
            assert (
                match_found
            ), f"Expected localisation {expected} not found in localized_modifications."

        # Ensure far masses are not tested (Oxidation and Phospho should be skipped)
        assert (
            mod_handler.get_localisation.call_count == 3
        )  # Should only be called 3 times because of cache

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

    def test_generate_modifications_combinations_lists_length_1(self, setup_modhandler_with_data):
        mod_handler = setup_modhandler_with_data

        # Test the function with combination_length=1
        masses, combinations = mod_handler.cache._generate_modifications_combinations_lists(
            combination_length=1
        )

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
        masses, combinations = mod_handler.cache._generate_modifications_combinations_lists(
            combination_length=2
        )

        # Expected results
        expected_masses = [
            10.0,  # mod1 (10)
            10.0 + 10.0,  # mod1 + mod1 (20)
            20.0,  # mod3 (20)
            10.0 + 20.0,  # mod1 + mod3 (30)
            40.0,  # mod2 (40)
            20.0 + 20.0,  # mod3 + mod3 (40)
            10.0 + 40.0,  # mod1 + mod2 (50)
            20.0 + 40.0,  # mod3 + mod2 (60)
            40.0 + 40.0,  # mod2 + mod2 (80)
        ]
        expected_combinations = [
            ("mod1",),  # 10
            ("mod1", "mod1"),  # 20
            ("mod3",),  # 20
            ("mod1", "mod3"),  # 30
            ("mod2",),  # 40
            ("mod3", "mod3"),  # 40
            ("mod1", "mod2"),  # 50
            ("mod3", "mod2"),  # 60
            ("mod2", "mod2"),  # 80
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
        masses, combinations = mod_handler.cache._generate_modifications_combinations_lists(
            combination_length=3
        )

        # Expected results
        expected_masses = [
            10.0,  # mod1 (10)
            10.0 + 10.0,  # mod1 + mod1 (20)
            20.0,  # mod3 (20)
            10.0 + 10.0 + 10.0,  # mod1 + mod1 + mod1 (30)
            10.0 + 20.0,  # mod1 + mod3 (30)
            20.0 + 20.0,  # mod3 + mod3 (40)
            10.0 + 10.0 + 20.0,  # mod1 + mod1 + mod3 (40)
            40.0,  # mod2 (40)
            10.0 + 40.0,  # mod1 + mod2 (50)
            10.0 + 20.0 + 20.0,  # mod1 + mod3 + mod3 (50)
            20.0 + 40.0,  # mod3 + mod2 (60)
            20.0 + 20.0 + 20.0,  # mod3 + mod3 + mod3 (60)
            10.0 + 10.0 + 40.0,  # mod1 + mod1 + mod2 (60)
            10.0 + 20.0 + 40.0,  # mod1 + mod3 + mod2 (70)
            40.0 + 40.0,  # mod2 + mod2 (80)
            20.0 + 20.0 + 40.0,  # mod3 + mod3 + mod2 (80)
            10.0 + 40.0 + 40.0,  # mod1 + mod2 + mod2 (90)
            20.0 + 40.0 + 40.0,  # mod3 + mod3 + mod2 (100)
            40.0 + 40.0 + 40.0,  # mod2 + mod2 + mod2 (120)
        ]

        expected_combinations = [
            ("mod1",),  # 10.0
            (
                "mod1",
                "mod1",
            ),  # 20.0
            ("mod3",),  # 20.0
            (
                "mod1",
                "mod1",
                "mod1",
            ),  # 30.0
            (
                "mod1",
                "mod3",
            ),  # 30.0
            (
                "mod3",
                "mod3",
            ),  # 40.0
            (
                "mod1",
                "mod1",
                "mod3",
            ),  # 40.0
            ("mod2",),  # 40.0
            (
                "mod1",
                "mod1",
                "mod3",
            ),  # 50.0
            (
                "mod1",
                "mod2",
            ),  # 50.0
            (
                "mod3",
                "mod2",
            ),  # 60.0
            (
                "mod3",
                "mod3",
                "mod3",
            ),  # 60.0
            (
                "mod1",
                "mod1",
                "mod2",
            ),  # 60.0
            (
                "mod1",
                "mod3",
                "mod2",
            ),  # 70.0
            (
                "mod2",
                "mod2",
            ),  # 80.0
            (
                "mod3",
                "mod3",
                "mod2",
            ),  # 80.0
            (
                "mod1",
                "mod2",
                "mod2",
            ),  # 90.0
            (
                "mod3",
                "mod2",
                "mod2",
            ),  # 100.0
            (
                "mod2",
                "mod2",
                "mod2",
            ),  # 120.0
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
        mod_handler.cache.modification_df = pd.DataFrame(columns=["name", "monoisotopic_mass"])

        masses, combinations = mod_handler.cache._generate_modifications_combinations_lists(
            combination_length=2
        )

        # Assertions
        assert masses == []
        assert combinations == []

    def test_double_combined_modifcations(self):

        mod_handler = _ModificationHandler(combination_length=2)

        psm = PSM(
            peptidoform="VTFTETPENGSKW/2",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
            precursor_mz="748.8581250320699",
        )

        localized_modifications = mod_handler.localize_mass_shift(psm)
        name_to_mass_dict = mod_handler.name_to_mass_residue_dict

        expmass = mz_to_mass(psm.precursor_mz, psm.get_precursor_charge())
        calcmass = calculate_mass(psm.peptidoform.composition)
        mass_shift = expmass - calcmass

        for candidate in localized_modifications:

            mass_shift1 = candidate.Localised_mass_shifts[0]

            # no need to check single mod 'combinations'
            try:
                mass_shift2 = candidate.Localised_mass_shifts[1]
            except:  # noqa: E722
                continue

            sum = (
                name_to_mass_dict[mass_shift1.modification].mass
                + name_to_mass_dict[mass_shift2.modification].mass
            )

            assert mass_shift1.loc != mass_shift2.loc
            assert sum >= (mass_shift - 0.02) and sum <= (mass_shift + 0.02)


if __name__ == "__main__":
    pytest.main()
