from Bio.PDB import PDBParser, PDBIO, Select
import os

class OneAltLocSelect(Select):
    def __init__(self):
        self.accepted_altlocs = dict()

    def accept_atom(self, atom):
        if not atom.is_disordered():
            return True

        altloc = atom.get_altloc()
        res_tuple = (atom.get_parent().get_id(), atom.get_parent().get_parent().get_id())

        if res_tuple not in self.accepted_altlocs:
            self.accepted_altlocs[res_tuple] = altloc
            return True
        else:
            return altloc == self.accepted_altlocs[res_tuple]

def remove_alternative_location_and_occupancy(input_pdb: str) -> None:
    input_basename = os.path.basename(input_pdb).replace(".pdb", "")
    input_dirname = os.path.dirname(input_pdb)
    with open(input_pdb, 'r') as infile, open(os.path.join(input_dirname, f'{input_basename}_noaltloc.pdb'), 'w') as outfile:
        for line in infile:
            if line.startswith('ATOM'):
                # Remove the 17th column (alternative location indicator) and shift everything to the left
                new_line1 = line[:16] +" "+ line[17:]
                # Set occupancy to 1.00 (columns 56-60)
                new_line = new_line1[:56] + "1.00" + new_line1[60:]
                outfile.write(new_line)
            else:
                outfile.write(line)
                

def keep_one_occupancy(input_pdb: str) -> None:
    parser = PDBParser(QUIET=True)
    input_basename = os.path.basename(input_pdb).replace(".pdb", "")
    input_dirname = os.path.dirname(input_pdb)
    
    structure = parser.get_structure("input", input_pdb)

    io = PDBIO()
    io.set_structure(structure)
    io.save(os.path.join(input_dirname, f'{input_basename}_oneocc.pdb'), OneAltLocSelect())
    
    remove_alternative_location_and_occupancy(os.path.join(input_dirname, f'{input_basename}_oneocc.pdb'))
