import os
import gzip
from Bio.PDB import MMCIFParser, PDBIO, Select
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from src.unmodify import unmodify_pdb


class ChainSelect(Select):
    def __init__(self, target_chain):
        self.target_chain = target_chain

    def accept_chain(self, chain):
        return chain.get_id() == self.target_chain
    
class CustomMMCIFParser(MMCIFParser):
    def _get_header(self):
        self.header = {}
        self._update_header_entry("name", ["_entry.id"])
        self._update_header_entry("head", ["_struct_keywords.pdbx_keywords"])
        self._update_header_entry("deposition_date", ["_pdbx_database_status.recvd_initial_deposition_date"])
        self._update_header_entry("release_date", ["_pdbx_database_status.SG_entry_release_date"])
        self._update_header_entry("structure_method", ["_exptl.method"])
        self._update_header_entry("resolution", ["_refine.ls_d_res_high", "_refine_hist.d_res_high"])

        try:
            self.header["resolution"] = float(self.header["resolution"])
        except (ValueError, KeyError):
            self.header["resolution"] = None

        return self.header
            

def process_chain(active_zerodisorder_name):
    # Directory containing IO files
    input_dir = './structures/mmcif_files_selection'
    output_dir = './structures/PDB_files_selection'
    parser = CustomMMCIFParser(QUIET=True)
    pdb_io = PDBIO()

    pdbid = active_zerodisorder_name.split("_")[4][:4].lower()
    chainid = active_zerodisorder_name.split("_")[4][4:]

    if len(chainid) > 1:
        return

    cif_gz_file = os.path.join(input_dir, f'{pdbid}.cif.gz')

    with gzip.open(cif_gz_file, 'rt') as cif_gz:
        structure = parser.get_structure(pdbid, cif_gz)

    pdb_file = os.path.join(output_dir, f'{active_zerodisorder_name}.pdb')
    pdbunmod_file = os.path.join(output_dir, f'{active_zerodisorder_name}_unmod.pdb')
    
    pdb_io.set_structure(structure)
    pdb_io.save(pdb_file, ChainSelect(chainid))
    unmodify_pdb(pdb_file, pdbunmod_file)
    os.remove(pdb_file)


