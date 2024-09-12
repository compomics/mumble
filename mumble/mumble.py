from copy import deepcopy
import logging
import itertools
from collections import namedtuple
from pathlib import Path

import pandas as pd
from psm_utils.io import read_file, write_file
from psm_utils import PSMList, PSM, Peptidoform
from psm_utils.utils import mz_to_mass
from pyteomics import proforma
from pyteomics.mass import std_aa_mass, calculate_mass, unimod
from pyteomics.fasta import IndexedFASTA
from rich.progress import track

logging.basicConfig(format="[%(asctime)s]%(levelname)s => %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)


class PSMHandler:
    """Class that contains all information about the input file"""

    def __init__(self, aa_combinations=0, fasta_file=None, mass_error=0.02, combination_length=1, exclude_mutations=False, **kwargs) -> None:
        """
        Constructor of the class.

        Args:
            input_file (str): Path to the input file
            filetype (str): Type of the input file to read with PSM_utlis.io.read_file
            aa_combinations (int, optional): Number of amino acid combinations to add as modification. Defaults to 0.
            fasta_file (str, optional): Path to the fasta file. Defaults to None.
            mass_error (float, optional): Mass error for the mass shift. Defaults to 0.02.
            exclude_mutations (bool, optional): If True, modifications with the classification 'AA substitution' will be excluded. Defaults to False.
        """

        # initialize modification seeker
        self.modification_handler = _ModificationHandler(
            mass_error=mass_error,
            add_aa_combinations=aa_combinations,
            fasta_file=fasta_file,
            combination_length=combination_length,
            exclude_mutations=exclude_mutations
        )
        self.psm_file_name = None

    @staticmethod
    def _find_mod_locations(peptidoform):
        """
        Find the locations of existing modifications in a peptide.

        Args:
            peptidoform (psm_utils.Peptidoform): Peptidoform object

        return:
            list: List of locations of existing modifications
        """
        locations = []

        if peptidoform.properties["n_term"] is not None:
            locations.append("N-term")

        if peptidoform.properties["c_term"] is not None:
            locations.append("C-term")

        for i, aa in enumerate(peptidoform.parsed_sequence):
            if aa[1] is not None:
                locations.append(i)

        return locations

    def _return_mass_shifted_peptidoform(self, modification_tuple_list, peptidoform) -> Peptidoform:
        """
        Apply a modification tuple to a peptidoform.

        Args:
            modification_tuple_list list(tuple): List of Modification_candidate namedtuple, containg a list of Localised_mass_shift nampedtuple
            peptidoform (psm_utils.Peptidoform): Peptidoform object

        return:
            psm_utils.Peptidoform: Peptidoform object
        """

        new_peptidoforms = []
        new_peptidoform = deepcopy(peptidoform)

        existing_mod_locations = self._find_mod_locations(new_peptidoform)
        
        for modification_tuple in modification_tuple_list:
            
            new_peptidoform = deepcopy(peptidoform)
            for localised_mass_shift in modification_tuple.Localised_mass_shifts:
                loc, mod = localised_mass_shift
                if loc in existing_mod_locations:
                    return None
                else:
                    if loc == "N-term":
                        new_peptidoform.properties["n_term"] = [proforma.process_tag_tokens(mod)]
                    elif loc == "C-term":
                        new_peptidoform.properties["c_term"] = [proforma.process_tag_tokens(mod)]
                    elif loc == "prepeptide":
                        new_peptidoform.parsed_sequence = [
                            (aa, None) for aa in mod
                        ] + new_peptidoform.parsed_sequence

                    elif loc == "postpeptide":
                        new_peptidoform.parsed_sequence = new_peptidoform.parsed_sequence + [
                            (aa, None) for aa in mod
                        ]
                    else:
                        try:
                            aa = new_peptidoform.parsed_sequence[loc][0]
                        except IndexError:
                            logger.warning(f"IndexError for {peptidoform} at {loc} with {mod}")
                            raise IndexError("Localisation is not in peptide")

                        # If the modification is an amino acid substitution
                        if mod in self.modification_handler.aa_sub_dict.keys():
                            if (
                                aa == self.modification_handler.aa_sub_dict[mod][0]
                            ):  # TODO named tuple so indexing is not necesary and more clear
                                new_peptidoform.parsed_sequence[loc] = (
                                    self.modification_handler.aa_sub_dict[mod][1],
                                    None,
                                )
                        # If the modification is a standard modification
                        else:
                            new_peptidoform.parsed_sequence[loc] = (
                                aa,
                                [proforma.process_tag_tokens(mod)],
                            )
            new_peptidoforms.append(new_peptidoform)
        return new_peptidoforms

    @staticmethod
    def _create_new_psm(psm, new_peptidoform) -> PSM:
        """
        Create new psm with new peptidoform.

        Args:
            psm (psm_utils.PSM): PSM object
            new_peptidoform (psm_utils.Peptidoform): Peptidoform object

        return:
            psm_utils.PSM: PSM object
        """
        if new_peptidoform is None:
            return
        copy_psm = deepcopy(psm)
        copy_psm.peptidoform = new_peptidoform
        return copy_psm

    def _get_modified_peptidoforms(self, psm, keep_original=False, warn=True) -> list:
        """
        Get modified peptidoforms derived from a single PSM.

        Args:
            psm (psm_utils.PSM): PSM object
            keep_original (bool, optional): Keep the original PSM. Defaults to False.
            warn (bool, optional): Warn if no modifications are found. Defaults to True.

        return:
            list: List of modified PSMs
        """
        modified_peptidoforms = []
        modification_tuple_list = self.modification_handler.localize_mass_shift(psm)
        if modification_tuple_list:
            new_proteoforms_list = self._return_mass_shifted_peptidoform(
                    modification_tuple_list, psm.peptidoform
            )
            for new_proteoform in new_proteoforms_list:
                new_psm = self._create_new_psm(
                        psm,
                        new_proteoform,
                )
                if new_psm is not None:
                    modified_peptidoforms.append(new_psm)
        elif warn:
            logger.warning(f"No modifications found for {psm}")
            return None
        if keep_original:
               modified_peptidoforms.append(psm)

        return modified_peptidoforms

    def get_modified_peptidoforms_list(self, psm, keep_original=False, warn=True) -> PSMList:
        """
        Get modified peptidoforms derived from 1 PSM in a PSMList.

        Args:
            psm (psm_utils.PSM): PSM object
            keep_original (bool, optional): Keep the original PSM. Defaults to False.
            warn (bool, optional): Warn if no modifications are found. Defaults to True.

        return:
            psm_utils.PSMList: PSMList object
        """
        modified_peptidoforms = self._get_modified_peptidoforms(
            psm, keep_original=keep_original, warn=warn
        )
        return PSMList(psm_list=modified_peptidoforms)

    def add_modified_psms(
        self, psm_list, psm_file_type="infer", generate_modified_decoys=False, keep_original=False
    ) -> PSMList:
        """
        Add modified psms to a psm list

        args:
            psm_list (str, list, PSMList): Path to the psm file, list of PSMs or PSMList object
            psm_file_type (str, optional): Type of the input file to read with PSM_utlis.io.read_file. Defaults to "infer" only used if psm_list is filepath.
            generate_modified_decoys (bool, optional): Generate modified decoys. Defaults to False.
            keep_original (bool, optional): Keep the original PSMs. Defaults to False.

        return:
            psm_utils.PSMList: PSMList object
        """

        logger.info(
            f"Adding modified PSMs to PSMlist {'WITH' if keep_original else 'WITHOUT'} originals, {'INCLUDING' if generate_modified_decoys else 'EXCLUDING'} modfied decoys"
        )

        parsed_psm_list = self.parse_psm_list(psm_list, psm_file_type)
        new_psm_list = []
        num_added_psms = 0

        for psm in track(
            parsed_psm_list,
            description="Parsing PSMs in PSMList...",
            total=len(parsed_psm_list),
        ):
            if (psm.is_decoy) & (not generate_modified_decoys):
                continue
            new_psms = self._get_modified_peptidoforms(
                psm, keep_original=keep_original, warn=False
            )
            if new_psms:
                num_added_psms += len(new_psms) if not keep_original else len(new_psms) - 1
                new_psm_list.extend(new_psms)
        if num_added_psms != 0:
            logger.info(f"Added {num_added_psms} additional modified PSMs")
        else:
            logger.warning("No modified PSMs found, ensure open modification search was enabled")

        return PSMList(psm_list=new_psm_list)

    def parse_psm_list(self, psm_list, psm_file_type="infer") -> PSMList:
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
        elif type(psm_list) is not PSMList:
            raise TypeError("psm_list should be a path to a file or a PSMList object")

        return psm_list

    def write_modified_psm_list(self, psm_list, output_file=None, psm_file_type="tsv"):
        """
        Write the modified PSM list to a file

        Args:
            psm_list (psm_utils.PSMList): PSMList object
            output_file (str, optional): Path to the output file. Defaults to None.
            psm_file_type (str, optional): Type of the output file to write with PSM_utlis.io.write_file. Defaults to "tsv".

        return:
            None
        """

        if self.psm_file_name and output_file is None:
            output_file = self.psm_file_name.stem + "_modified"

        elif not self.psm_file_name and output_file is None:
            logger.warning("No output file specified")
            output_file = "modified_psm_list"

        logger.info(f"Writing modified PSM list to {output_file}")
        write_file(psm_list=psm_list, filename=output_file, filetype=psm_file_type)

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
        
        

class _ModificationHandler:
    """Class that handles modifications."""

    def __init__(
        self,
        mass_error=0.02,
        add_aa_combinations=0,
        fasta_file=None,
        combination_length=1,
        exclude_mutations=False
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
        self.combination_length = combination_length
        self.get_unimod_database(exclude_mutations)
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

    def get_unimod_database(self, exclude_mutations=False):
        """
        Read unimod databse to a dataframe.
        
        Args:
            exclude_mutations (bool, optional): If True, modifications with the classification 'AA substitution' will be excluded. Defaults to False.
        """
        unimod_db = unimod.Unimod()
        # if necesary, make distinction protein and peptide level C-term and N-term modifications
        position_id_mapper = {
            2: "anywhere",
            3: "N-term",
            4: "C-term",
            5: "N-term",
            6: "C-term",
        }

        modifications = []
        for mod in unimod_db.mods:
            if (
                not mod.username_of_poster == "unimod"
            ):  # Do not include user submitted modifications
                continue
            name = mod.ex_code_name
            if not name:
                name = mod.code_name
            if ("Xlink" in name) or ("plex" in name):  # Do not include crosslinks
                continue
            monoisotopic_mass = mod.monoisotopic_mass
            for specificity in mod.specificities:
                classification = specificity.classification
                if classification == "Isotopic label":  # Do not include isotopic labels
                    continue
                if exclude_mutations and classification.classification == "AA substitution":
                    continue 
                position = specificity.position_id
                aa = specificity.amino_acid
                modifications.append(
                    {
                        "name": name,
                        "monoisotopic_mass": monoisotopic_mass,
                        "classification": classification.classification,
                        "restriction": position_id_mapper[position],
                        "residue": aa,
                        "rounded_mass": round(monoisotopic_mass, 0),
                    }
                )

        self.modification_df = pd.DataFrame(
            modifications,
            columns=[
                "name",
                "monoisotopic_mass",
                "classification",
                "restriction",
                "residue",
                #"rounded_mass",
            ],
        )
        
        self.monoisotopic_masses, self.modifications_names = self._generate_modifications_combinations_lists(self.combination_length)
        


    def _generate_modifications_combinations_lists(self, combination_length=1):
        """
        Generates all possible combinations of modifications and calculates their summed monoisotopic masses.

        This method creates combinations of modifications up to the specified combination length and calculates
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
        modification_filtered_df = self.modification_df[['name', 'monoisotopic_mass']].drop_duplicates()

        # Initial setup: combinations with just one modification
        modifications = [(item,) for item in modification_filtered_df['name']]
        monoisotopic_masses = modification_filtered_df['monoisotopic_mass'].to_list()

        # Iteratively build combinations for length > 1
        for r in range(2, combination_length + 1):
            new_modifications = []
            new_monoisotopic_masses = []

            # Iterate through the previous set of modifications and add new modifications
            for i in range(len(modifications)):
                for j in range(len(modification_filtered_df)):
                    new_comb = modifications[i] + (modification_filtered_df.iloc[j]['name'],)
                    # Ensure the combination is sorted for consistency and to avoid duplicates
                    sorted_comb = tuple(sorted(new_comb))

                    # Only add if it's unique (no repeated combinations)
                    if sorted_comb not in new_modifications:
                        new_modifications.append(sorted_comb)
                        new_monoisotopic_masses.append(monoisotopic_masses[i] + modification_filtered_df.iloc[j]['monoisotopic_mass'])

            # Update the modifications and masses lists with the new combinations
            modifications.extend(new_modifications)
            monoisotopic_masses.extend(new_monoisotopic_masses)

        # Combine masses and modifications into tuples, sort by monoisotopic mass, then unzip back into two lists
        combined = sorted(zip(monoisotopic_masses, modifications))

        if combined:
            monoisotopic_masses, modifications = zip(*combined)
            return list(monoisotopic_masses), list(modifications)
        else:
            return [], []

    def _get_name_to_mass_residue_dict(self):
        """
        Get dictionary with name as key and mass and residue as value

        return:
            dict: Dictionary with name as key and mass and residue as value
        """
        Modification = namedtuple("modification", ["mass", "residues", "restrictions"])

        return {
            row.name: Modification(row.monoisotopic_mass, row.residue, row.restriction)
            for row in self.modification_df.groupby(["monoisotopic_mass", "name"])
            .agg({"residue": list, "restriction": list})
            .reset_index()
            .itertuples()
        }  # TODO: used named tuple here

    def get_localisation(
        self, psm, modification_name, residue_list, restrictions
    ) -> set[namedtuple]:
        """
        Localise a given modification in a peptide

        Args:
            psm (psm_utils.PSM): PSM object
            modification_name (str): Name of the modification
            residue_list (list): List of residues
            restrictions (list): List of restrictions

            return:
                list: List of localised_mass_shift
        """
        loc_list = []
        Localised_mass_shift = namedtuple("Localised_mass_shift", ["loc", "modification"])

        amino_acids_peptide = [x[0] for x in psm.peptidoform.parsed_sequence]

        for residue, restriction in zip(residue_list, restrictions):
            if (residue == "N-term") and (psm.peptidoform.properties["n_term"] is None):
                loc_list.append(Localised_mass_shift("N-term", modification_name))

            elif residue == "C-term" and (psm.peptidoform.properties["c_term"] is None):
                loc_list.append(Localised_mass_shift("C-term", modification_name))

            elif residue == "protein_level":
                loc_list.extend(
                    [
                        Localised_mass_shift(loc, mod)
                        for loc, mod in self.check_protein_level(psm, modification_name)
                    ]
                )
            elif restriction == "N-term" or restriction == "C-term":
                if (
                    restriction == "N-term"
                    and (psm.peptidoform.properties["n_term"] is None)
                    and (psm.peptidoform.parsed_sequence[0][0] == residue)
                ):
                    loc_list.append(Localised_mass_shift("N-term", modification_name))

                elif (
                    restriction == "C-term"
                    and (psm.peptidoform.properties["c_term"] is None)
                    and (psm.peptidoform.parsed_sequence[-1][0] == residue)
                ):
                    loc_list.append(Localised_mass_shift("C-term", modification_name))

                else:
                    continue

            elif residue in amino_acids_peptide:
                loc_list.extend(
                    [
                        Localised_mass_shift(i, modification_name)
                        for i, aa in enumerate(amino_acids_peptide)
                        if (aa == residue) and (psm.peptidoform.parsed_sequence[i][1] is None)
                    ]
                )

        # remove duplicate locations
        return set(loc_list)

    def localize_mass_shift(self, psm) -> list[namedtuple]:
        """Give potential localizations of a mass shift in a peptide

        Args:
            psm (psm_utils.PSM): PSM object

        return:
            list: List of Modification_candidate([localised_mass_shift])
        """

        expmass = mz_to_mass(psm.precursor_mz, psm.get_precursor_charge())
        calcmass = calculate_mass(psm.peptidoform.composition)
        mass_shift = expmass - calcmass

        # get all potential modifications
        try:
            potential_modifications_indices = self._binary_range_search(self.monoisotopic_masses, mass_shift, self.mass_error)
            if potential_modifications_indices:
                potential_modifications_tuples = self.modifications_names[
                    potential_modifications_indices[0]:potential_modifications_indices[1] + 1]
            else:
                return []

        except KeyError:
            return None

        Localised_mass_shift = namedtuple("Localised_mass_shift", ["loc", "modification"])
        Modification_candidate = namedtuple("Modification_candidate", ["Localised_mass_shifts"])

        feasible_modifications_candidates = []

        # Memoization cache for storing feasible subcombinations
        memo_cache = {}

        def check_subcombinations(combination):
            """Check if all subcombinations of the given combination are feasible."""
            subcombination_feasible = True
            subcombinations_data = []
            
            for subcombination in itertools.combinations(combination, len(combination) - 1):
                sorted_subcomb = tuple(sorted(subcombination))
                if sorted_subcomb in memo_cache:
                    # Get previously computed localization data for the subcombination
                    subcombinations_data.append(memo_cache[sorted_subcomb])
                else:
                    subcombination_feasible = False
                    break
            
            return subcombination_feasible, subcombinations_data

        for potential_mods_combination in potential_modifications_tuples:
            sorted_combination = tuple(sorted(potential_mods_combination))
            
            # Skip combinations larger than the max length and don't store them in memo
            if len(sorted_combination) > self.max_combination_length:
                continue

            # If combination is already stored in memo, use that
            if sorted_combination in memo_cache:
                feasible = memo_cache[sorted_combination]["feasible"]
                if not feasible:
                    continue  # Skip if already known to be infeasible
            else:
                # Check subcombinations to see if they are feasible
                subcombination_feasible, subcombinations_data = check_subcombinations(sorted_combination)
                if not subcombination_feasible:
                    continue  # Skip if any subcombination is infeasible

                # Compute localization for the current combination
                localized_modification_positions = []
                feasible = True

                for mod_name in potential_mods_combination:
                    residues = self.name_to_mass_residue_dict[mod_name].residues
                    restrictions = self.name_to_mass_residue_dict[mod_name].restrictions
                    localized_mods = self.get_localisation(psm, mod_name, residues, restrictions)

                    if localized_mods:
                        localized_modification_positions.append(localized_mods)
                    else:
                        feasible = False
                        break

                # Store the current combination in memo if it's feasible
                if feasible and len(sorted_combination) <= self.max_combination_length:
                    memo_cache[sorted_combination] = {
                        "feasible": True,
                        "positions": localized_modification_positions
                    }

            # Generate all possible position combinations for the modifications
            possible_combinations = list(itertools.product(*localized_modification_positions))

            # Filter out combinations where two modifications would occupy the same position
            valid_combinations = []
            for combination in possible_combinations:
                positions = [mod.loc for mod in combination]
                if len(set(positions)) == len(positions):  # No overlap in positions
                    valid_localized_mods = [Localised_mass_shift(loc=mod.loc, modification=mod.modification) for mod in combination]
                    feasible_modifications_candidates.append(Modification_candidate(Localised_mass_shifts=valid_localized_mods))

            # If the combination is feasible and less than max combination length, store it in memo
            if len(sorted_combination) < self.max_combination_length:
                memo_cache[sorted_combination] = {
                    "feasible": feasible,
                    "positions": localized_modification_positions
                }

        return feasible_modifications_candidates if feasible_modifications_candidates else None


    def _binary_range_search(self, arr, target, error) -> tuple[int,int]:
        """
        Find the indexes of values within a specified range in a ascending array.

        Args:
            arr (list of int/float): A sorted array in ascending order.
            target (int/float): The midpoint value defining the center of the range.
            error (int/float): The acceptable deviation from the target, defining the size of the range.

        Returns:
            tuple: A tuple containing the start and end indexes of the values that fall within the range 
            target - error, target + error]. If no values are found, returns an empty tuple.
        """

        def binary_left_index(arr, value) -> int:
            '''
            Finds the index of the smallest element in a sorted array that is greater than or equal to a given value.
            '''

            left, right= 0,len(arr)-1
            result = len(arr)

            while left <= right:
                mid = (left+right) // 2 # round to int

                if arr[mid] >= value:
                    result = mid
                    right = mid - 1
                else:
                    left = mid + 1
            return result

        def binary_right_index(arr, value) -> int:
            '''
            Finds the index of the biggest element in a sorted array that is less than or equal to a given value.
            '''

            left, right = 0,len(arr)-1
            result = len(arr)

            while left <= right:
                mid = (left+right) // 2 # round to int

                if arr[mid] <= value:
                    result = mid
                    left = mid + 1
                else:
                    right = mid - 1
            return result

        left = binary_left_index(arr,target-error)
        right = binary_right_index(arr,target+error)

        if left <= right:
            return (left, right)
        else:
            return ()

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
                        "rounded_mass": round(mass, 0),
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
        found_additional_amino_acids = []

        protein_sequence = self.fasta_file[psm.protein_list[0]].sequence
        peptide_start_position = protein_sequence.find(psm.peptidoform.sequence)
        peptide_end_position = peptide_start_position + len(psm.peptidoform.sequence)
        additional_aa_len = len(additional_aa)

        if (
            protein_sequence[peptide_start_position - additional_aa_len : peptide_start_position]
            == additional_aa
        ):
            found_additional_amino_acids.append(("prepeptide", additional_aa))

        if (
            protein_sequence[peptide_end_position : peptide_end_position + additional_aa_len]
            == additional_aa
        ):
            found_additional_amino_acids.append(("postpeptide", additional_aa))

        return found_additional_amino_acids
