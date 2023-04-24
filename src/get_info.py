import os
import re

def get_gene_group(file_path):
    filename = os.path.basename(file_path)
    return "_".join(filename.split("_")[:2])
    
def get_conf(file_path):
    filename = os.path.basename(file_path)
    conf = re.search(r'(conf\d+)', filename)
    if conf:
        return conf.group(1)
    return None


# Load the PDB files from the first folder into a dictionary with lists of file paths as values
def get_pdb_files(pdb_folder1, pdb_folder2, get_gene_group):
    # Load the AF2 PDB files from the first folder into a dictionary
    pdb_files1 = {}
    for file in os.listdir(pdb_folder1):
        if file.endswith(".pdb"):
            gene_group = get_gene_group(file)
            file_path = os.path.join(pdb_folder1, file)
            if gene_group in pdb_files1:
                pdb_files1[gene_group].append(file_path)
            else:
                pdb_files1[gene_group] = [file_path]

    # Load the RCSB PDB files from the benchmark folder into a dictionary
    pdb_files2 = {}
    for file in os.listdir(pdb_folder2):
        if file.endswith(".pdb"):
            gene_group = get_gene_group(file)
            file_path = os.path.join(pdb_folder2, file)
            if gene_group in pdb_files2:
                pdb_files2[gene_group].append(file_path)
            else:
                pdb_files2[gene_group] = [file_path]
                
    return pdb_files1, pdb_files2


# Function to extract "Active" or "Inactive" from the file name
def get_status(filename):
    if "_Active_" in filename:
        return "Active"
    elif "_Inactive_" in filename:
        return "Inactive"
    else:
        return "Unknown"

# gets first chain from pdb structure Biopython
def get_first_chain_id(structure):
    return next(structure[0].get_chains()).get_id()
    

def get_tmalign_data(output_str):
    output_lines = output_str.splitlines()

    # Extract first sequence
    first_sequence = output_lines[-4]

    # Extract second sequence
    second_sequence = output_lines[-2]

    # Extract TM-scores
    tm_scores_match = re.search(r'TM-score=\s*([\d.]+).*\nTM-score=\s*([\d.]+)', output_str)
    tm_score_1 = float(tm_scores_match.group(1)) if tm_scores_match else None
    tm_score_2 = float(tm_scores_match.group(2)) if tm_scores_match else None

    # Extract RMSD
    rmsd_match = re.search(r'RMSD=\s*([\d.]+)', output_str)
    rmsd = float(rmsd_match.group(1)) if rmsd_match else None

    return {
        "first_sequence": first_sequence,
        "second_sequence": second_sequence,
        "tm_score_1": tm_score_1,
        "tm_score_2": tm_score_2,
        "rmsd": rmsd,
    }



