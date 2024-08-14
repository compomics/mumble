import pytest
from unittest.mock import MagicMock
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
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep=True)

        assert isinstance(new_psms, list)
        assert len(new_psms) == 2
        assert new_psms[0].peptidoform.properties["n_term"] == ["Acetyl"]
        assert new_psms[1] == psm

        mod_handler.localize_mass_shift.return_value = [(1, "Carbamyl"), (4, "Carbamyl")]
        new_psms = psm_handler._get_modified_peptidoforms(psm, keep=False)

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


class TestModificationHandler:

    @pytest.fixture
    def setup_modhandler(self):
        # Fixture for setting up _ModificationHandler
        psm = PSM(
            peptidoform="ART[Deoxy]HR/3",
            spectrum_id="some_spectrum",
            is_decoy=False,
            protein_list=["some_protein"],
        )
        mod_handler = _ModificationHandler(mass_error=0.02)
        return mod_handler, psm

    def test_get_unimod_database(self, setup_modhandler):
        mod_handler = setup_modhandler
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

    def test_localize_mass_shift(self, setup_modhandler):
        mod_handler, psm = setup_modhandler

        mod_handler.name_to_mass_residue_dict = {
            "Carbamyl": namedtuple("Modification", ["mass", "residues", "restrictions"])(
                43.005814, ["C", "R"], ["anywhere", "anywhere"]
            ),
            "Acetyl": namedtuple("Modification", ["mass", "residues", "restrictions"])(
                42.010565, ["N-term"], ["any N-term"]
            ),
        }
        mod_handler.rounded_mass_to_name_dict = {43.0: ["Carbamyl"], 42.0: ["Acetyl"]}

        psm.precursor_mz = 214.126106 + (43.005814 / 3)
        localized_modifications = mod_handler.localize_mass_shift(psm)
        assert localized_modifications is not None
        assert localized_modifications[0] == (1, "Carbamyl")
        assert localized_modifications[1] == (4, "Carbamyl")

        psm.precursor_mz = 214.126106 + (42.010565 / 3)
        localized_modifications = mod_handler.localize_mass_shift(psm)
        assert localized_modifications is not None
        assert localized_modifications[0] == ("N-term", "Acetyl")

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


if __name__ == "__main__":
    pytest.main()
