import os
#from pymol import cmd
from pymol2 import PyMOL
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.b_factor import activation_loop_average_and_min_b_factor, overall_average_b_factor, calculate_b_factors
from src.activation_loop import parse_activation_loops
from src.get_info import get_gene_group, get_pdb_files, get_status, get_first_chain_id
import datetime

# ... (your existing code to initialize variables and load PDB files) ...
# Define the structures folder
structures_folder = os.path.abspath("./structures")
# PDB folders within the structures folder
pdb_folder1 = os.path.join(structures_folder, "AF220_PDB_230317")
pdb_folder2 = os.path.join(structures_folder, "PDB_ATP_sele")


#parser = PDBParser()
activation_loops = parse_activation_loops('actloop.fasta')
pdb_files1, pdb_files2 = get_pdb_files(pdb_folder1, pdb_folder2, get_gene_group)



# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns= ["Gene_Group", "PDB_File1", "PDB_File2",
                                    "RMSD_Before_Alignment", "RMSD_After_Alignment_actloop40N", 
                                    "RMSD_After_Cealignment_actloop40N", "RMSD_After_Super_actloop40N", 
                                    "RMSD_After_Alignment_whole", "RMSD_After_Cealignment_whole", "RMSD_After_Super_whole",
                                    "Avg_PLDDT_All", "Min_PLDDT_All", "Avg_PLDDT_Act_Loop", "Min_PLDDT_Act_Loop", "Status"])


def process_gene_group(gene_group, pdb_file1_list, pdb_files2, activation_loops):
    results = []
    if gene_group in pdb_files2:
        pdb_file2 = pdb_files2[gene_group]
        for pdb_file1 in pdb_file1_list:
            start_residue, end_residue = map(int, activation_loops[gene_group].split('-'))

            # Adjust the start_residue to include 40 residues before
            start_residue_adjusted = max(1, start_residue - 40)

            # Parse the structures
            parser = PDBParser()
            structure1 = parser.get_structure('struct1', pdb_file1)
            structure2 = parser.get_structure('struct2', pdb_file2)

            # Get the chain IDs
            chain_id1 = get_first_chain_id(structure1)
            chain_id2 = get_first_chain_id(structure2)

            # Get the last residue numbers
            last_residue1 = max(residue.id[1] for residue in structure1[0][chain_id1] if residue.id[0] == ' ' and not residue.is_disordered())
            last_residue2 = max(residue.id[1] for residue in structure2[0][chain_id2] if residue.id[0] == ' ' and not residue.is_disordered())

            # Set the last_residue number to the minimum of the last residues of both proteins
            last_residue_common = min(last_residue1, last_residue2)

            # Get the starting residue numbers
            start_residue1 = min(residue.id[1] for residue in structure1[0][chain_id1] if residue.id[0] == ' ' and not residue.is_disordered())
            start_residue2 = min(residue.id[1] for residue in structure2[0][chain_id2] if residue.id[0] == ' ' and not residue.is_disordered())

            # Set the start_residue number to the maximum of the starting residues of both proteins
            start_residue_common = max(start_residue1, start_residue2)

            with PyMOL() as pymol:
                cmd = pymol.cmd

                # Load the structures into PyMOL
                cmd.load(pdb_file1, f"{gene_group}_1")
                cmd.load(pdb_file2, f"{gene_group}_2")
                
                # Define the selection strings for the residue range
                actloop_selection1 = f"{gene_group}_1 and chain {chain_id1} and resi {start_residue}-{end_residue} and name n+ca+c+o"
                actloop_selection2 = f"{gene_group}_2 and chain {chain_id2} and resi {start_residue}-{end_residue} and name n+ca+c+o"

                # Define the selection strings for the residue range
                N40actloop_selection1 = f"{gene_group}_1 and chain {chain_id1} and resi {start_residue_adjusted}-{last_residue_common} and name n+ca+c+o"
                N40actloop_selection2 = f"{gene_group}_2 and chain {chain_id2} and resi {start_residue_adjusted}-{last_residue_common} and name n+ca+c+o"
                
                # Define the selection strings for the whole protein with common start_residue
                whole_protein_selection1 = f"{gene_group}_1 and chain {chain_id1} and resi {start_residue_common}-{last_residue_common} and name n+ca+c+o"
                whole_protein_selection2 = f"{gene_group}_2 and chain {chain_id2} and resi {start_residue_common}-{last_residue_common} and name n+ca+c+o"
                
                
                # calculate RMSD and perform total alignment and calculate RMSD over the activation loop again
                rmsd_before_align = cmd.rms_cur(actloop_selection1, actloop_selection2, matchmaker=-1)
                
                cmd.delete("all"), cmd.load(pdb_file1, f"{gene_group}_1"), cmd.load(pdb_file2, f"{gene_group}_2")
                alignment = cmd.align(N40actloop_selection1, N40actloop_selection2)
                rmsd_after_align_actloop40N = cmd.rms_cur(actloop_selection1, actloop_selection2, matchmaker=-1)
                
                cmd.delete("all"), cmd.load(pdb_file1, f"{gene_group}_1"), cmd.load(pdb_file2, f"{gene_group}_2")
                alignment = cmd.cealign(N40actloop_selection1, N40actloop_selection2)
                rmsd_after_cealign_actloop40N = cmd.rms_cur(actloop_selection1, actloop_selection2, matchmaker=-1)
                
                cmd.delete("all"), cmd.load(pdb_file1, f"{gene_group}_1"), cmd.load(pdb_file2, f"{gene_group}_2")
                alignment = cmd.super(N40actloop_selection1, N40actloop_selection2)
                rmsd_after_super_actloop40N = cmd.rms_cur(actloop_selection1, actloop_selection2, matchmaker=-1)
                
                cmd.delete("all"), cmd.load(pdb_file1, f"{gene_group}_1"), cmd.load(pdb_file2, f"{gene_group}_2")
                alignment = cmd.align(whole_protein_selection1, whole_protein_selection2)
                rmsd_after_align_whole = cmd.rms_cur(actloop_selection1, actloop_selection2, matchmaker=-1)
                
                cmd.delete("all"), cmd.load(pdb_file1, f"{gene_group}_1"), cmd.load(pdb_file2, f"{gene_group}_2")
                alignment = cmd.cealign(whole_protein_selection1, whole_protein_selection2)
                rmsd_after_cealign_whole = cmd.rms_cur(actloop_selection1, actloop_selection2, matchmaker=-1)
                
                cmd.delete("all"), cmd.load(pdb_file1, f"{gene_group}_1"), cmd.load(pdb_file2, f"{gene_group}_2")
                alignment = cmd.super(whole_protein_selection1, whole_protein_selection2)
                rmsd_after_super_whole = cmd.rms_cur(actloop_selection1, actloop_selection2, matchmaker=-1)
                
                # Calculate and print B-factors
                b_factors = calculate_b_factors(pdb_file1, chain_id1, start_residue, end_residue)

                # Get the status (Active or Inactive)
                status = get_status(pdb_file1)
              
                # Append the results to the DataFrame
                new_row = {
                    "Gene_Group": gene_group,
                    "PDB_File1": pdb_file1,
                    "PDB_File2": pdb_file2,
                    "RMSD_Before_Alignment": rmsd_before_align,
                    "RMSD_After_Alignment_actloop40N": rmsd_after_align_actloop40N,
                    "RMSD_After_Cealignment_actloop40N": rmsd_after_cealign_actloop40N,
                    "RMSD_After_Super_actloop40N": rmsd_after_super_actloop40N,
                    "RMSD_After_Alignment_whole": rmsd_after_align_whole,
                    "RMSD_After_Cealignment_whole": rmsd_after_cealign_whole,
                    "RMSD_After_Super_whole": rmsd_after_super_whole,
                    "Avg_PLDDT_All": b_factors[0],
                    "Min_PLDDT_All": b_factors[1],
                    "Avg_PLDDT_Act_Loop": b_factors[2],
                    "Min_PLDDT_Act_Loop": b_factors[3],
                    "Status": status
                }
                results.append(new_row)
                
                
                # Delete the structures from PyMOL
                cmd.delete("all")

    return results


### Main loop
with ProcessPoolExecutor() as executor:
    # Start the processes
    futures = [executor.submit(process_gene_group, gene_group, pdb_file1_list, pdb_files2, activation_loops) for gene_group, pdb_file1_list in list(pdb_files1.items())[:100]]

    # Collect the results as they become available
    for future in tqdm(as_completed(futures), total=len(futures)):
        results_df = pd.concat([results_df, pd.DataFrame(future.result(), columns=results_df.columns)], ignore_index=True)
