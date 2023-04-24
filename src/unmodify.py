import os

def unmodify_pdb(input_pdb: str, output_pdb: str) -> None:
    modified_residues = {
        'SEP': 'SER',
        'PTR': 'TYR',
        'TPO': 'THR',
    }

    extra_atoms = {'P', 'O1P', 'O2P', 'O3P'}

    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb}")

    with open(input_pdb, 'r') as f_in, open(output_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                res_name = line[17:20]
                atom_name = line[12:16].strip()
                if res_name in modified_residues and atom_name in extra_atoms:
                    continue
                elif res_name in modified_residues and atom_name not in extra_atoms:
                    unmodified_res_name = modified_residues[res_name]
                    line = 'ATOM  ' + line[6:17] + unmodified_res_name + line[20:]
                    f_out.write(line)
                else:
                    f_out.write(line)
            else:
                f_out.write(line)

