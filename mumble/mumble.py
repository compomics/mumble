from copy import deepcopy
import logging
import itertools
import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from itertools import product

import pandas as pd
import pickle
from psm_utils.io import read_file, write_file
from psm_utils import PSMList, PSM, Peptidoform
from psm_utils.utils import mz_to_mass
from pyteomics import proforma
from pyteomics.mass import std_aa_mass, calculate_mass, unimod
from pyteomics.fasta import IndexedFASTA
from rich.progress import track
from rich.logging import RichHandler

# setup logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(message)s",  # Simple format for logging
    datefmt="[%X]",  # Time format
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
)

# Add a logger
logger = logging.getLogger("rich")


class PSMHandler:
    """Class that contains all information about the input file"""

    def __init__(self, config_file: str = None, **kwargs):
        """
        Initialize PSMHandler.

        Args:
            config_file (str, optional): Path to the JSON configuration file. Defaults to None.
            **kwargs: Additional configurations to override file-based settings.
        """
        self.config_loader = JSONConfigLoader(config_file) if config_file else None
        self.params = self._load_parameters(kwargs)

        # initialize modification seeker
        self.modification_handler = _ModificationHandler(
            mass_error=self.params["mass_error"],
            add_aa_combinations=self.params["aa_combinations"],
            fasta_file=self.params["fasta_file"],
            combination_length=self.params["combination_length"],
            exclude_mutations=self.params["exclude_mutations"],
            unimod_modification_file=self.params["unimod_modification_file"],
        )
        self.psm_file_name = None

    def _load_parameters(self, overrides: dict) -> dict:
        """
        Load configuration parameters from the JSON file and apply overrides.

        Args:
            overrides (dict): Dictionary of parameters to override.

        Returns:
            dict: Consolidated configuration parameters.
        """
        keys_with_defaults = {
            "mass_error": 0.02,
            "combination_length": 1,
            "exclude_mutations": False,
            "aa_combinations": 0,
            "fasta_file": None,
            "psm_list": None,
            "output_file": None,
            "write_filetype": "tsv",
            "keep_original": False,
            "generate_modified_decoys": False,
            "psm_file_type": "infer",
            "unimod_modification_file": None,
            "modification_mapping": {},
        }

        # Use a single loop to consolidate parameters
        params = {
            key: overrides.get(
                key, self.config_loader.get(key, default) if self.config_loader else default
            )
            for key, default in keys_with_defaults.items()
        }
        logger.info(f"Mumble config: {params}")

        return params

    def _return_mass_shifted_psms(self, modification_lists, psm) -> Peptidoform:
        """
        Apply a list of modification tuples to a peptidoform.

        Args:
            modification_tuple_list (list of tuples): List of modification tuples containing local mass shifts.
            peptidoform (psm_utils.Peptidoform): Original peptidoform object to modify.

        Returns:
            list of psm_utils.Peptidoform: List of new peptidoform objects with applied modifications, or None if conflicting modifications exist.
        """
        mass_shifted_psms = []

        for assigned_modifications in modification_lists:

            mass_shifted_psm = deepcopy(psm)
            new_peptidoform = mass_shifted_psm.peptidoform  # TODO check if link is broken

            # handle terminal modifications
            if nterm := assigned_modifications[0]:
                new_peptidoform.properties["n_term"] = [proforma.process_tag_tokens(nterm)]
            if cterm := assigned_modifications[-1]:
                new_peptidoform.properties["c_term"] = [proforma.process_tag_tokens(cterm)]

            # handle peptide modifications
            for i, mod in enumerate(assigned_modifications[2:-2]):
                if mod in self.modification_handler.aa_sub_dict.keys():
                    new_peptidoform.parsed_sequence[i] = (
                        self.modification_handler.aa_sub_dict[mod][1],
                        None,
                    )
                elif mod:
                    new_peptidoform.parsed_sequence[i] = (
                        new_peptidoform.parsed_sequence[i][0],
                        [proforma.process_tag_tokens(mod)],
                    )

            # handle peptide additions
            if additional_aa := assigned_modifications[1]:
                new_peptidoform.parsed_sequence = [
                    (aa, None) for aa in additional_aa
                ] + new_peptidoform.parsed_sequence
            if additional_aa := assigned_modifications[-2]:
                new_peptidoform.parsed_sequence = new_peptidoform.parsed_sequence + [
                    (aa, None) for aa in additional_aa
                ]

            mass_shifted_psms.append(mass_shifted_psm)

        return mass_shifted_psms

    def _get_modified_peptidoforms(self, psm, keep_original=False) -> list:
        """
        Get modified peptidoforms derived from a single PSM.

        Args:
            psm (psm_utils.PSM): Original PSM object.
            keep_original (bool, optional): Whether to keep the original PSM alongside modified ones. Defaults to False.

        Returns:
            list: List of modified PSMs, or None if no modifications were applied.
        """
        modified_peptidoforms = []

        if keep_original:
            psm["metadata"]["original_psm"] = True
            modified_peptidoforms.append(psm)

        localised_mass_shifts = self.modification_handler._localize_mass_shift(psm)
        if localised_mass_shifts:
            modified_peptidoforms.append(
                self._return_mass_shifted_psms(localised_mass_shifts, psm)
            )

        return modified_peptidoforms

    def get_modified_peptidoforms_list(self, psm, keep_original=False) -> PSMList:
        """
        Get modified peptidoforms derived from 1 PSM in a PSMList.

        Args:
            psm (psm_utils.PSM): PSM object
            keep_original (bool, optional): Keep the original PSM. Defaults to False.

        return:
            psm_utils.PSMList: PSMList object
        """
        modified_peptidoforms = self._get_modified_peptidoforms(psm, keep_original=keep_original)
        return PSMList(psm_list=modified_peptidoforms)

    def add_modified_psms(
        self,
        psm_list=None,
        psm_file_type=None,
        generate_modified_decoys=None,
        keep_original=None,
    ) -> PSMList:
        """
        Add modified PSMs to a PSMList based on open modification searches.

        Args:
            psm_list (str, list, or PSMList): Path to a PSM file, list of PSMs, or a PSMList object.
            psm_file_type (str, optional): Type of PSM file to read, inferred automatically if not provided. Defaults to "infer".
            generate_modified_decoys (bool, optional): Whether to generate decoys for the modified PSMs. Defaults to False.
            keep_original (bool, optional): Whether to keep the original unmodified PSMs. Defaults to False.

        Returns:
            psm_utils.PSMList: A new PSMList object containing the modified PSMs.
        """
        if not psm_list:
            if psm_list := self.params["psm_list"]:
                pass
            else:
                raise ValueError("No PSM list provided")
        if not generate_modified_decoys:
            generate_modified_decoys = self.params["generate_modified_decoys"]
        if not keep_original:
            keep_original = self.params["keep_original"]
        if not psm_file_type:
            psm_file_type = self.params["psm_file_type"]

        logger.info(
            f"Adding modified PSMs to PSMlist {'WITH' if keep_original else 'WITHOUT'} originals, {'INCLUDING' if generate_modified_decoys else 'EXCLUDING'} modfied decoys"
        )

        parsed_psm_list = self._parse_psm_list(
            psm_list, psm_file_type, self.params["modification_mapping"]
        )
        new_psm_list = []
        num_added_psms = 0

        for psm in track(
            parsed_psm_list,
            description="Matching modifications to your mass shifts ... ",
            total=len(parsed_psm_list),
        ):
            if (psm.is_decoy) & (not generate_modified_decoys):
                continue
            new_psms = self._get_modified_peptidoforms(psm, keep_original=keep_original)
            if new_psms:
                num_added_psms += len(new_psms) if not keep_original else len(new_psms) - 1
                new_psm_list.extend(new_psms)
        if num_added_psms != 0:
            logger.info(f"Added {num_added_psms} additional modified PSMs")
        else:
            logger.warning("No modified PSMs found, ensure open modification search was enabled")

        return PSMList(psm_list=new_psm_list)

    def _parse_psm_list(
        self, psm_list, psm_file_type="infer", modification_mapping=dict()
    ) -> PSMList:
        """
        Parse the psm list to get the peptidoform and protein information

        Args:
            psm_list (str, list, PSMList): Path to the psm file, list of PSMs or PSMList object
            psm_file_type (str, optional): Type of the input file to read with PSM_utlis.io.read_file. Defaults to "infer".

        return:
            psm_utils.PSMList: PSMList object
        """

        if type(psm_list) is PSMList:
            pass
        elif type(psm_list) is list:
            psm_list = PSMList(psm_list=psm_list)
        elif type(psm_list) is str:
            self.psm_file_name = Path(psm_list)
            psm_list = read_file(psm_list, filetype=psm_file_type)
            psm_list.rename_modifications(modification_mapping)
        elif type(psm_list) is not PSMList:
            raise TypeError("psm_list should be a path to a file or a PSMList object")

        return psm_list

    def write_modified_psm_list(self, psm_list, output_file=None, psm_file_type=None):
        """
        Write the modified PSM list to a file

        Args:
            psm_list (psm_utils.PSMList): PSMList object
            output_file (str, optional): Path to the output file. Defaults to None.
            psm_file_type (str, optional): Type of the output file to write with PSM_utlis.io.write_file. Defaults to "tsv".

        return:
            None
        """
        if not output_file:
            if output_file := self.params["output_file"]:
                pass

            elif self.psm_file_name:
                output_file = self.psm_file_name.stem + "_modified"

            elif not self.psm_file_name:
                logger.warning("No output file specified")
                output_file = "modified_psm_list"

        if not psm_file_type:
            psm_file_type = self.params["write_filetype"]

        logger.info(f"Writing modified PSM list to {output_file}")
        write_file(psm_list=psm_list, filename=output_file, filetype=psm_file_type)


class _ModificationHandler:
    """Class that handles modifications."""

    def __init__(
        self,
        mass_error=0.02,
        add_aa_combinations=0,
        fasta_file=None,
        combination_length=1,
        exclude_mutations=False,
        unimod_modification_file=None,
    ) -> None:
        """
        Constructor of the class.

        Args:
            mass_error (float, optional): Mass error for the mass shift. Defaults to 0.02.
            add_aa_combinations (int, optional): Number of amino acid combinations to add as modification. Defaults to 0.
            fasta_file (str, optional): Path to the fasta file. Defaults to None.
            combination_length (int, optional): Maximum number of modifications per combination. All lower numbers will be included as well. Dfeaults to 1.
            exclude_mutations (bool, optional): If True, modifications with the classification 'AA substitution' will be excluded. Defaults to False.
        """
        # TODO add amino acid variations (mutation) as flag
        self.cache = ModificationCache(
            combination_length=combination_length,
            exclude_mutations=exclude_mutations,
            modification_file=unimod_modification_file,
        )

        self.modification_df = self.cache.modification_df
        self.monoisotopic_masses = self.cache.monoisotopic_masses
        self.modifications_names = self.cache.modifications_names

        logger.info(
            f'Including {len(self.modification_df["name"].unique())} unique modifications on {len(self.modification_df["name"])} sites'
        )

        if add_aa_combinations:
            if not fasta_file:
                raise ValueError("Fasta file is required to add amino acid combinations")
            self._add_amino_acid_combinations(add_aa_combinations)
            self.protein_level_check = True
        else:
            self.protein_level_check = False

        self.name_to_mass_residue_dict = self._get_name_to_mass_residue_dict()
        self.aa_sub_dict = self._get_aa_sub_dict()

        self.mass_error = mass_error
        self.fasta_file = IndexedFASTA(fasta_file, label=r"^[\n]?>([\S]*)") if fasta_file else None

    def _get_name_to_mass_residue_dict(self):
        """
        Get dictionary with name as key and mass and residue as value

        return:
            dict: Dictionary with name as key and mass and residue as value
        """

        return {
            row.name: Modification(
                name=row.name,
                monoisotopic_mass=row.monoisotopic_mass,
                residues=row.residue,
                restrictions=row.restriction,
            )
            for row in self.modification_df.groupby(["monoisotopic_mass", "name"])
            .agg({"residue": list, "restriction": list})
            .reset_index()
            .itertuples()
        }

    def _localize_mass_shift(self, psm):
        """Give potential localisations of a mass shift in a peptide

        Args:
            psm (psm_utils.PSM): PSM object

        return:
            list: List of Modification_candidate([localised_mass_shift])
        """
        expmass = mz_to_mass(psm.precursor_mz, psm.get_precursor_charge())
        calcmass = calculate_mass(
            psm.peptidoform.composition
        )  # TODO: use mass instead of composition
        mass_shift = expmass - calcmass

        peptidofrom_as_list = (
            (["N-term"] if not psm.peptidoform.properties["n_term"] else [None])
            + (["Protein_level"] if self.protein_level_check else [None])
            + ([x[0] if not x[1] else None for x in psm.peptidoform.parsed_sequence])
            + (["Protein_level"] if self.protein_level_check else [None])
            + (["C-term"] if not psm.peptidoform.properties["c_term"] else [None])
        )

        # get all potential modifications
        search_indices = self._binary_range_search(
            self.monoisotopic_masses, mass_shift, self.mass_error
        )
        if not search_indices:
            return []
        canididate_mass_shifts = self.modifications_names[
            search_indices[0] : search_indices[1] + 1
        ]
        psm_modifications = []
        for modifications in canididate_mass_shifts:
            # Try localise all modifications for each mass shift candidate
            peptidoform_candidates = []
            for mod_name in modifications:
                localised_mod = self.name_to_mass_residue_dict[mod_name].localise(
                    peptidofrom_as_list
                )
                if localised_mod[1] or localised_mod[-2]:
                    localised_mod[1], localised_mod[-2] = self.check_protein_level(psm, mod_name)

                if not localised_mod.any():
                    break

                peptidoform_candidates.append(localised_mod)
            if peptidoform_candidates:
                final_candidates = self._generate_combinations(peptidoform_candidates)
                psm_modifications.extend(final_candidates)

        return psm_modifications

    @staticmethod
    def _generate_combinations(mod_lists):
        """Give a list of lists of single modification placements, generate all potential combinations"""
        mod_array = np.array(mod_lists, dtype=object)

        # Use np.where to get indices of non-False values for each row
        valid_indices = [
            np.where(mod_array[i] != False)[0].tolist() for i in range(mod_array.shape[0])
        ]
        # generate all possible combinations of indices
        valid_combinations = [
            comb for comb in list(product(*valid_indices)) if len(set(comb)) == len(mod_lists)
        ]

        candidates = []
        # Generate the candidate modifications
        for combination in valid_combinations:
            candidate = [None] * len(mod_array[0])
            for i, idx in enumerate(combination):
                name = mod_array[i][idx]
                if candidate[idx] == "[B]":
                    candidate = False
                    break

                elif "[B]" in name:
                    candidate[idx] = name.replace("[B]", "")
                    # check that there are no other modifications on before this position
                    if any(candidate[:idx]):
                        candidate = False
                        break
                    else:
                        # block everything before this position
                        candidate[:idx] = ["[B]"] * idx
                else:
                    candidate[idx] = name
            if candidate:
                # replace [B] with None
                candidate = [None if x == "[B]" else x for x in candidate]
                candidates.append(candidate)

        return candidates

    @staticmethod
    def _binary_range_search(arr, target, error) -> tuple[int, int]:
        """
        Finds the indexes of values within a specified range in a sorted array.

        Args:
            arr (list of int/float): A sorted array in ascending order.
            target (int/float): The midpoint value defining the center of the range.
            error (int/float): The acceptable deviation from the target, defining the size of the range.

        Returns:
            tuple: A tuple containing the start and end indexes of the values that fall within the range
                [target - error, target + error]. If no values are found, returns an empty tuple.
        """
        if not arr:
            return ()

        def binary_left_index(arr, value) -> int:
            """
            Finds the index of the smallest element in a sorted array that is greater than or equal to a given value.
            """
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] >= value:
                    right = mid - 1
                else:
                    left = mid + 1
            return left

        def binary_right_index(arr, value) -> int:
            """
            Finds the index of the largest element in a sorted array that is less than or equal to a given value.
            """
            left, right = 0, len(arr) - 1
            while left <= right:
                mid = (left + right) // 2
                if arr[mid] <= value:
                    left = mid + 1
                else:
                    right = mid - 1
            return right

        lower_bound = target - error
        upper_bound = target + error

        left = binary_left_index(arr, lower_bound)
        right = binary_right_index(arr, upper_bound)

        # Check bounds and validity
        if left <= right and left < len(arr) and right >= 0:
            return (left, right)
        return None

    def _get_aa_sub_dict(self):
        """
        Get dictionary with name as key and mass and residue as value.

        return:
            dict: Dictionary with name as key and mass and residue as value
        """

        aa_sub_df = self.modification_df[
            self.modification_df["classification"] == "AA substitution"
        ]
        aa_dict = {
            name.split("->")[0]: residue
            for name, residue in zip(aa_sub_df["name"], aa_sub_df["residue"])
        }
        aa_sub_dict = {
            name: (residue, aa_dict[name.split("->")[1]])
            for name, residue in zip(aa_sub_df["name"], aa_sub_df["residue"])
        }
        return aa_sub_dict

    def _add_amino_acid_combinations(self, number_of_aa=1):
        """
        Add amino acid masses to the modification dataframe

        Args:
            number_of_aa (int, optional): Number of amino acids to add. Defaults to 1.
        """
        aa_combinations = []
        for n in range(1, number_of_aa + 1):
            aa_combinations.extend(list(itertools.product("ACDEFGHIKLMNPQRSTVWY", repeat=n)))
        aa_to_mass_dict = {
            "".join(combo): sum([round(std_aa_mass[aa], 6) for aa in combo])
            for combo in aa_combinations
        }
        self.modification_df = pd.concat(
            [
                self.modification_df,
                pd.DataFrame(
                    {
                        "name": name,
                        "monoisotopic_mass": mass,
                        "classification": "AA addition",
                        "residue": "protein_level",
                        "restriction": "anywhere",
                    }
                    for name, mass in aa_to_mass_dict.items()
                ),
            ]
        )

    def check_protein_level(self, psm, additional_aa):
        """
        Check if amino acid(s) precedes or follows a peptide in the protein sequence.

        Args:
            psm (psm_utils.PSM): PSM object
            additional_aa (str): Additional amino acid to check
        """

        # Do this for decoys? Then we should be able to reverese sequences and shuffled decoys will never work
        # TODO Multiple proteins require PSMs to be split based on proteins

        if psm.is_decoy:
            return []

        protein_sequence = self.fasta_file[psm.protein_list[0]].sequence
        peptide_start_position = protein_sequence.find(psm.peptidoform.sequence)
        peptide_end_position = peptide_start_position + len(psm.peptidoform.sequence)
        additional_aa_len = len(additional_aa)

        if (
            protein_sequence[peptide_start_position - additional_aa_len : peptide_start_position]
            == additional_aa
        ):
            prepeptide = additional_aa
        else:
            prepeptide = False

        if (
            protein_sequence[peptide_end_position : peptide_end_position + additional_aa_len]
            == additional_aa
        ):
            postpeptide = additional_aa
        else:
            postpeptide = False

        return prepeptide, postpeptide


class ModificationCache:
    """Class that handles the cache for modifications."""

    def __init__(
        self, combination_length=1, exclude_mutations=False, modification_file=None
    ) -> None:
        """
        Constructor of the class.

        Args:
            combination_length (int, optional): Maximum number of modifications per combination. All lower numbers will be included as well. Defaults to 1.
            exclude_mutations (bool, optional): If True, modifications with the classification 'AA substitution' will be excluded. Defaults to False.
        """
        self.combination_length = combination_length
        self.exclude_mutations = exclude_mutations
        self.modification_file = modification_file
        self.modification_inclusion_dict, self.filter_key = self._read_unimod_file(
            modification_file
        )
        self.monoisotopic_masses = []
        self.modifications_names = []
        self.modification_df = None

        # Load or generate data
        cache_file = self._get_cache_file_path()
        self._load_or_generate_data(cache_file)

    def _get_cache_file_path(self):
        """
        Get path to cache file for combinations of modifications.

        return:
            str: path to cache file
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(current_dir)
        cache_dir = os.path.join(parent_dir, "modification_cache")

        # Create the cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, "modification_cache.pkl")
        return cache_file

    def _load_or_generate_data(self, cache_file: str) -> None:
        """Load data from cache or generate and save it if cache doesn't exist."""
        if os.path.exists(cache_file):
            logger.info("Checking cache")
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            if cache_data["metadata"] == (
                self.combination_length,
                self.exclude_mutations,
                self.modification_file,
            ):
                try:
                    logger.info("Loading cache data")
                    self.modification_df = cache_data["modification_df"]
                    self.monoisotopic_masses = cache_data["monoisotopic_masses"]
                    self.modifications_names = cache_data["modifications_names"]
                except KeyError:
                    logger.info("Cache data missing")
                    self._regenerate_and_save_cache(cache_file)
            else:
                self._regenerate_and_save_cache(cache_file)
        else:
            self._regenerate_and_save_cache(cache_file)

    def get_unimod_database(self):
        """
        Read Unimod database to a DataFrame.

        Args:
            exclude_mutations (bool, optional): If True, modifications with the classification 'AA substitution' will be excluded. Defaults to False.
        """
        unimod_db = unimod.Unimod()
        position_id_mapper = {
            2: "anywhere",
            3: "N-term",
            4: "C-term",
            5: "N-term",
            6: "C-term",
        }

        modifications = []
        for mod in unimod_db.mods:
            # Get modification name and ID
            name = mod.ex_code_name or mod.code_name
            unimod_id = mod.id

            # Filter based on custom inclusion dictionary
            if self.modification_inclusion_dict:
                key = unimod_id if self.filter_key == "unimod_id" else name
                if key not in self.modification_inclusion_dict.keys():
                    continue

            # Default filtering for Unimod database
            elif mod.username_of_poster != "unimod" or any(
                term in name.lower() for term in ["xlink", "plex", "label"]
            ):
                continue

            monoisotopic_mass = mod.monoisotopic_mass

            for specificity in mod.specificities:
                classification = specificity.classification.classification

                # Skip based on classification
                if classification == "Isotopic label":
                    continue
                if self.exclude_mutations and classification == "AA substitution":
                    continue

                position = specificity.position_id
                aa = specificity.amino_acid

                # Additional filtering based on inclusion dictionary
                if self.modification_inclusion_dict:
                    aa_list = self.modification_inclusion_dict.get(
                        unimod_id if self.filter_key == "unimod_id" else name
                    )
                    if aa_list and aa not in aa_list:
                        continue

                # Append modification details
                modifications.append(
                    {
                        "name": name,
                        "monoisotopic_mass": monoisotopic_mass,
                        "classification": classification,
                        "restriction": position_id_mapper.get(position, "unknown"),
                        "residue": aa,
                    }
                )

        # Create a DataFrame from the modifications
        self.modification_df = pd.DataFrame(
            modifications,
            columns=[
                "name",
                "monoisotopic_mass",
                "classification",
                "restriction",
                "residue",
            ],
        )

    def _generate_modifications_combinations_lists(self, combination_length=1):
        """
        Generates all possible combinations of modifications and calculates their summed monoisotopic masses.
        This method creates unique combinations of modifications up to the specified combination length and calculates
        the total monoisotopic mass for each combination. The results are returned as two lists: one containing
        the summed monoisotopic masses and the other containing the corresponding modification combinations.
        The results are sorted by the summed monoisotopic masses in ascending order.

        Args:
        combination_length (int, optional): Maximum number of modifications per combination. All lower numbers will be included as well. Defaults to 1.

        Returns:
        tuple: A tuple containing two elements:
            - List[float]: A list of summed monoisotopic masses, sorted in ascending order.
            - List[tuple]: A list of tuples, where each tuple contains a combination of modification names
                        corresponding to the summed monoisotopic masses.
        """
        # Remove duplicates from the modification DataFrame
        modification_filtered_df = self.modification_df[
            ["name", "monoisotopic_mass"]
        ].drop_duplicates()
        mass_dict = dict(
            zip(modification_filtered_df["name"], modification_filtered_df["monoisotopic_mass"])
        )

        # Function to generate combinations
        def generate_combinations(items, length):
            if length == 0:
                yield ()
            elif length > 0:
                for i in range(len(items)):
                    for cc in generate_combinations(items[i:], length - 1):
                        yield (items[i],) + cc

        # Generate all unique combinations
        all_modifications = []
        all_masses = []
        unique_combinations = set()

        for r in range(1, combination_length + 1):
            for combo in generate_combinations(sorted(modification_filtered_df["name"]), r):
                if combo not in unique_combinations:
                    unique_combinations.add(combo)
                    all_modifications.append(combo)
                    all_masses.append(sum(mass_dict[mod] for mod in combo))

        # Sort the results by mass
        combined = sorted(zip(all_masses, all_modifications))

        if combined:
            monoisotopic_masses, modifications = zip(*combined)
            return list(monoisotopic_masses), list(modifications)
        else:
            return [], []

    def _regenerate_and_save_cache(self, cache_file: str) -> None:
        """Regenerate data and save it to the cache."""
        logger.info("Generating cache data")
        self.get_unimod_database()
        self.monoisotopic_masses, self.modifications_names = (
            self._generate_modifications_combinations_lists(self.combination_length)
        )

        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "metadata": (
                        self.combination_length,
                        self.exclude_mutations,
                        self.modification_file,
                    ),
                    "modification_df": self.modification_df,
                    "monoisotopic_masses": self.monoisotopic_masses,
                    "modifications_names": self.modifications_names,
                },
                f,
            )

        # Update class variables
        self.monoisotopic_masses = self.monoisotopic_masses
        self.modifications_names = self.modifications_names
        self.modification_df = self.modification_df

    def _read_unimod_file(self, modification_file=None):
        """
        Read a list of modifications from a file.

        Args:
            modification_file (str, optional): Path to the modification file. Defaults to None.

        Returns:
            list: List of modifications
        """

        if modification_file:
            df = pd.read_csv(modification_file, sep="\t")
            req_columns = ["unimod_id", "name"]
            for key in req_columns:
                if key in df.columns:
                    if "residue" in df.columns:
                        grouped = df.groupby([key]).agg({"residue": list}).reset_index()
                        return {row[key]: row["residue"] for _, row in grouped.iterrows()}, key
                    else:
                        return {row[key]: None for _, row in df.iterrows()}, key
            raise ValueError("Modification file should contain 'id' or 'name' column")
        else:
            return None, None


@dataclass
class Modification:
    name: str
    monoisotopic_mass: float
    residues: np.ndarray
    restrictions: np.ndarray

    def __post_init__(self):
        assert len(self.residues) == len(
            self.restrictions
        ), "Residues and restrictions should have the same length"

    def localise(self, peptidoform_as_list):
        """
        Localise the modification in a peptide based on residues and restrictions.

        Args:
            peptidoform_as_list (np.ndarray): Array of the original peptidoform.

        Returns:
            np.ndarray: Array with modifcation name, [B]Modification name or False for each element.
        """
        # convert to array
        peptidoform_as_list = np.array(peptidoform_as_list)
        # Initialize result array with False
        result = np.full(peptidoform_as_list.shape, False, dtype=object)

        # Precompute positions
        n_term_positions = np.array([0, 2])
        c_term_positions = np.array([len(peptidoform_as_list) - 1, len(peptidoform_as_list) - 3])

        for residue, restriction in zip(self.residues, self.restrictions):
            # Find matches
            matches = peptidoform_as_list == residue

            if restriction == "anywhere":
                result[matches] = self.name

            elif restriction == "N-term":
                # Handle N-term restrictions
                n_term_indices = np.intersect1d(np.where(matches)[0], n_term_positions)
                result[n_term_indices] = "[B]" + self.name
                result[np.where(matches)[0][~np.isin(np.where(matches)[0], n_term_positions)]] = (
                    False
                )
                result[np.where(matches)[0][np.isin(np.where(matches)[0], [0])]] = self.name

            elif restriction == "C-term":
                # Handle C-term restrictions
                c_term_indices = np.intersect1d(np.where(matches)[0], c_term_positions)
                result[c_term_indices] = "[B]" + self.name
                result[np.where(matches)[0][~np.isin(np.where(matches)[0], c_term_positions)]] = (
                    False
                )
                result[
                    np.where(matches)[0][
                        np.isin(np.where(matches)[0], [len(peptidoform_as_list) - 1])
                    ]
                ] = self.name

        return result


class JSONConfigLoader:
    """Loads a single-level configuration from a JSON file."""

    def __init__(self, config_file: str):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file: str):
        with open(config_file, "r") as f:
            return json.load(f)

    def get(self, key: str, default=None):
        """Retrieve a configuration value."""
        return self.config.get(key, default)
