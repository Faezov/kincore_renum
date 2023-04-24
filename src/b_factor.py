# Functions related to B-factor calculations
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB import PDBParser
parser = PDBParser()

def activation_loop_average_and_min_b_factor(structure, chain_id, start_residue, end_residue):
    residues = structure[0][chain_id]
    b_factors = []

    for residue in residues:
        residue_id = residue.get_id()[1]
        if start_residue <= residue_id <= end_residue:
            try:
                atom = residue['CA']
                b_factors.append(atom.get_bfactor())
            except KeyError:
                pass
            
    return [sum(b_factors) / len(b_factors), min(b_factors)]

def overall_average_b_factor(structure, chain_id):
    residues = structure[0][chain_id]
    b_factors = []
    
    for residue in residues:
        try:
            atom = residue['CA']
            b_factors.append(atom.get_bfactor())
        except KeyError:
            pass
        
    return sum(b_factors) / len(b_factors)

# B-factor calculation function
def calculate_b_factors(pdb_file, chain_id, start_residue, end_residue):
    # Parse the PDB file
    try:
        structure = parser.get_structure("structure", pdb_file)
    except PDBConstructionException:
        return (0, 0, 0, 0)

    # Calculate the average and minimum B-factor for all CA atoms
    avg_b_factor_all = overall_average_b_factor(structure, chain_id)
    min_b_factor_all = min(atom.get_bfactor() for atom in structure.get_atoms() if atom.get_name() == "CA")

    # Calculate the average and minimum B-factor for CA atoms in the activation loop
    avg_b_factor_act_loop, min_b_factor_act_loop = activation_loop_average_and_min_b_factor(structure, chain_id, start_residue, end_residue)

    # Print the results
    return [avg_b_factor_all, min_b_factor_all, avg_b_factor_act_loop, min_b_factor_act_loop]


