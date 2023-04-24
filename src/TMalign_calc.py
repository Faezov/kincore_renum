import os
import re
import subprocess
from Bio.PDB import PDBParser
from pymol2 import PyMOL
from src.get_info import get_gene_group, get_pdb_files, get_first_chain_id, get_tmalign_data
from src.unmodify import unmodify_pdb
from src.clean_up_files import clean_up_files


def TMalign_calc(structure1_path, structure2_path, activation_loops):
    # Get the file name without extension
    structure1_name = os.path.splitext(os.path.basename(structure1_path))[0]
    structure2_name = os.path.splitext(os.path.basename(structure2_path))[0]
    # Create the output path for the unmodified PDB
    structure2_unmodified_path = os.path.join(os.path.dirname(structure2_path), structure2_name + '_unmodified.pdb')

    # Call the unmodify_pdb function
    unmodify_pdb(structure2_path, structure2_unmodified_path)

    group_gene = get_gene_group(structure1_path)
    output_prefix = f"aligned_whole_{structure1_name}"

    start_residue, end_residue = map(int, activation_loops[group_gene].split('-'))
    # Adjust the start_residue to include 40 residues before
    start_residue_adjusted = max(1, start_residue - 40)

    # Parse the structures
    parser = PDBParser()
    structure1 = parser.get_structure('struct1', structure1_path)
    structure2 = parser.get_structure('struct2', structure2_unmodified_path)

    # Get the chain IDs
    chain_id1 = get_first_chain_id(structure1)
    chain_id2 = get_first_chain_id(structure2)

    # Get the last residue numbers
    last_residue1 = max(residue.id[1] for residue in structure1[0][chain_id1] if residue.id[0] == ' ' and not residue.is_disordered())
    last_residue2 = max(residue.id[1] for residue in structure2[0][chain_id2] if residue.id[0] == ' ' and not residue.is_disordered())
    # Set the last_residue number to the minimum of the last residues of both proteins
    last_residue_common = min(last_residue1, last_residue2)

    with PyMOL() as pymol:
        cmd = pymol.cmd
        # Load the structures into PyMOL
        cmd.load(structure1_path, f"{group_gene}_1")
        cmd.load(structure2_unmodified_path, f"{group_gene}_2")

        # Define the selection strings for the residue range
        N40actloop_selection1 = f"{group_gene}_1 and chain {chain_id1} and resi {start_residue_adjusted}-{last_residue_common}"
        N40actloop_selection2 = f"{group_gene}_2 and chain {chain_id2} and resi {start_residue_adjusted}-{last_residue_common}"

        command = ["TMalign", structure1_path, structure2_unmodified_path, "-o", output_prefix]
        output_str = subprocess.run(command, capture_output=True, text=True).stdout
        tmalign_whole_data = get_tmalign_data(output_str)

        os.rename(f"{output_prefix}_all_atm", f"{output_prefix}_all_atm.pdb")
        cmd.load(f"{output_prefix}_all_atm.pdb", "TMaligned")

        # Define the selection strings for the residue range
        selection1 = f"TMaligned and chain A and resi {start_residue}-{end_residue} and name n+ca+c+o"
        selection2 = f"TMaligned and chain B and resi {start_residue}-{end_residue} and name n+ca+c+o"
        # Calculate the RMSD
        RMSD_After_TMalign_whole = cmd.rms_cur(selection1, selection2, matchmaker=-1)

        cmd.delete("all")
        cmd.load(structure1_path, f"{group_gene}_1")
        cmd.load(structure2_unmodified_path, f"{group_gene}_2")

        # Define the selection strings for the residue range
        pymol.cmd.select(f"{structure1_name}_N40actloop_selection1", N40actloop_selection1)
        pymol.cmd.save(f"{structure1_name}_N40actloop_selection1.pdb", f"{structure1_name}_N40actloop_selection1")
        pymol.cmd.select(f"{structure1_name}_N40actloop_selection2", N40actloop_selection2)
        pymol.cmd.save(f"{structure1_name}_N40actloop_selection2.pdb", f"{structure1_name}_N40actloop_selection2")
        output_prefix = f"aligned_N40actloop_{structure1_name}"

        command = ["TMalign", f"{structure1_name}_N40actloop_selection1.pdb", f"{structure1_name}_N40actloop_selection2.pdb", "-o", output_prefix]
        output_str = subprocess.run(command, capture_output=True, text=True).stdout
        tmalign_N40actloop_data = get_tmalign_data(output_str)

        os.rename(f"{output_prefix}_all_atm", f"{output_prefix}_all_atm.pdb")
        cmd.load(f"{output_prefix}_all_atm.pdb", "TMaligned")

        #Calculate the RMSD
        RMSD_After_TMalign_actloop40N = cmd.rms_cur(selection1, selection2, matchmaker=-1)

        clean_up_files(structure1_name)
        cmd.delete("all")
        
        TMalign_dict = {"RMSD_After_TMalign_actloop40N": RMSD_After_TMalign_actloop40N,
                        "RMSD_After_TMalign_whole": RMSD_After_TMalign_whole,
                        "tmalign_whole_data": tmalign_whole_data,
                        "tmalign_N40actloop_data": tmalign_N40actloop_data,}
        
        return TMalign_dict
        
        
# ### usage
# from src.TMalign_calc import TMalign_calc
# from src.activation_loop import parse_activation_loops
# from src.get_info import get_pdb_files,get_gene_group
# import os

# # ... (your existing code to initialize variables and load PDB files) ...
# # Define the structures folder
# structures_folder = os.path.abspath("./structures")
# # PDB folders within the structures folder
# pdb_folder1 = os.path.join(structures_folder, "AF220_PDB_230317")
# pdb_folder2 = os.path.join(structures_folder, "PDB_ATP_sele")


# #parser = PDBParser()
# activation_loops = parse_activation_loops('actloop.fasta')
# pdb_files1, pdb_files2 = get_pdb_files(pdb_folder1, pdb_folder2, get_gene_group)

# for gene_group, pdb_file1_list in list(pdb_files1.items())[:4]:
#     if gene_group in pdb_files2:
#         pdb_file2 = pdb_files2[gene_group]
#         for pdb_file1 in pdb_file1_list:
#             TMalign_data = TMalign_calc(pdb_file1, pdb_file2, activation_loops)
