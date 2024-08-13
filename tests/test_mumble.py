import pytest
from unittest.mock import MagicMock
from collections import namedtuple
from psm_utils import PSMList, PSM, Peptidoform
from pyteomics import proforma
from mumble.mumble import _ModificationHandler, PSMHandler


class TestPSMHandler:

    @pytest.fixture
    def setup_psmhandler(self):
        # Fixture for setting up PSMHandler with mocked dependencies
        psm = MagicMock(spec=PSM)
        psm.peptidoform = Peptidoform("ART[Deoxy]HR")
        psm.is_decoy = False

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
        assert new_peptidoform_2.parsed_sequence == [
            ("A", None),
            ("R", None),
            ("T", "Deoxy"),
            ("A", None),
            ("R", None),
        ]

    def test_create_new_psm(self, setup_psmhandler):
        psm_handler, _, psm = setup_psmhandler

        new_peptidoform = MagicMock()
        new_peptidoform.parsed_sequence = [
            ("A", None),
            ("R", None),
            ("T", "Deoxy"),
            ("A", None),
            ("R", None),
        ]
        new_psm = psm_handler._create_new_psm(psm, new_peptidoform)

        assert new_psm is not None
        assert new_psm.peptidoform == new_peptidoform

    def test_get_modified_peptidoforms(self, setup_psmhandler):
        psm_handler, mod_handler, psm = setup_psmhandler

        mod_handler.aa_sub_dict = {"His->Ala": ("H", "A")}

        mod_handler.localize_mass_shift.return_value = [("N-term", "Acetyl")]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep=True)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 2
        assert new_psms[0].peptidoform.properties["n_term"] == ["Acetyl"]
        assert new_psms[1] == psm

        mod_handler.localize_mass_shift.return_value = [(1, "Carbamyl"), (4, "Carbamyl")]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep=False)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 2
        assert new_psms[0].peptidoform.parsed_sequence[1] == ("R", "Carbamyl")
        assert new_psms[1].peptidoform.parsed_sequence[4] == ("R", "Carbamyl")

    def test_add_modified_psms(self, setup_psmhandler):
        psm_handler, mod_handler, psm = setup_psmhandler

        psm_list = [psm]
        mod_handler.localize_mass_shift.return_value = [("N-term", "mod1")]
        new_psm_list = psm_handler.add_modified_psms(psm_list)

        assert isinstance(new_psm_list, PSMList)
        assert len(new_psm_list) > 1


class TestModificationHandler:

    @pytest.fixture
    def setup_modhandler(self):
        # Fixture for setting up _ModificationHandler
        mod_handler = _ModificationHandler(mass_error=0.02)
        return mod_handler

    def test_get_unimod_database(self, setup_modhandler):
        mod_handler = setup_modhandler
        mod_handler.get_unimod_database()

        assert mod_handler.modification_df is not None

    def test_localize_mass_shift(self, setup_modhandler):
        mod_handler = setup_modhandler

        psm = MagicMock(spec=PSM)
        psm.peptidoform = MagicMock()
        psm.peptidoform.parsed_sequence = [("A", None), ("C", None), ("D", None)]
        psm.peptidoform.properties = {"n_term": None, "c_term": None}
        psm.precursor_mz = 500.0
        psm.get_precursor_charge.return_value = 2

        mod_handler.name_to_mass_residue_dict = {
            "mod1": namedtuple("Modification", ["mass", "residues", "restrictions"])(
                57.0, ["C"], ["anywhere"]
            )
        }
        mod_handler.rounded_mass_to_name_dict = {57.0: ["mod1"]}

        localized_modifications = mod_handler.localize_mass_shift(psm)
        assert localized_modifications is not None

    def test_check_protein_level(self, setup_modhandler):
        mod_handler = setup_modhandler

        mod_handler.fasta_file = MagicMock()
        mod_handler.fasta_file.__getitem__.return_value.sequence = "ACDEFGHIKLMNPQRSTVWY"

        psm = MagicMock(spec=PSM)
        psm.peptidoform.sequence = "CDEFGH"
        psm.protein_list = ["some_protein"]
        additional_aa = "A"

        results = mod_handler.check_protein_level(psm, additional_aa)
        assert ("prepeptide", additional_aa) in results


if __name__ == "__main__":
    pytest.main()
