import os
import re
import shutil
import gzip
from pathlib import Path
import Bio
from Bio.PDB import *
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
pd.set_option('display.max_rows', 1000)

from src.download_data import url_formation_for_pool, download_with_pool
from src.renum.mmCIF.new_mmCIFv2 import *
exception_AccessionIDs = ["P42212", "Q17104", "Q27903", "Q93125", "P03069", 
                          "D3DLN9", "Q96UT3", "P0ABE7", "P00192", "P76805", 
                          "Q8XCE3", "P00720", "Q38170", "Q94N07", "P0AEX9", 
                          "P02928", "Q2M6S0"]

default_input_path_to_mmCIF="./mmCIF"
default_input_path_to_SIFTS="./SIFTS"
default_output_path_to_mmCIF="./mmCIF_renum"                          
if not os.path.exists(default_output_path_to_mmCIF):
    os.makedirs(default_output_path_to_mmCIF)


default_num=50000
gzip_mode="on"



with open("active_zerodisorder.txt", "r") as active_zerodisorder:
    active_zerodisorder_rl = active_zerodisorder.readlines()
    active_zerodisorder_names = [f.strip().replace(".cif", "") for f in active_zerodisorder_rl]
    active_pdbfn = [f.strip().split("_")[4][:4].lower()+".cif" for f in active_zerodisorder_rl]


def renum_mmCIF(mmCIF):
    log_message = list()
    mmCIF_name = mmCIF[:4] + ".cif.gz"
    SIFTS_name = mmCIF_name[:4] + ".xml.gz"
    
    mmcif_dict = try_mmCIF2dict(default_input_path_to_mmCIF, mmCIF_name)
    PDBe_PDB, PDBe_UniProt_AccessionID, UniProt_conversion_dict = try_SIFTS_tree_parser(default_input_path_to_SIFTS="./SIFTS", SIFTS_name=SIFTS_name)

    # _no UniProt in SIFTS _no_UniProt_in_SIFTS_out.cif.gz
    if PDBe_UniProt_AccessionID == list():
        copy_file(default_input_path_to_mmCIF, mmCIF_name, default_output_path_to_mmCIF, ".cif.gz", gzip_mode)
        log_message = if_no_SIFTS_data_log(mmCIF_name, mmcif_dict, log_message)
        

    df_PDBe_PDB_UniProt, df_PDBe_PDB_UniProt_WOnull = make_df_from_SIFTS_data(PDBe_PDB, PDBe_UniProt_AccessionID, 
                                                                              default_num, chains_to_change="all")
    
    chains_to_change, chains_to_change_1toN, AccessionIDs, chain_AccessionID_dict = get_chains_and_accessions(df_PDBe_PDB_UniProt)
    combined_PDBe_UniProt_AccessionID, longest_AccessionIDs = resolve_numbering_clashes(df_PDBe_PDB_UniProt, exception_AccessionIDs, 
                                                                                        chain_AccessionID_dict)
    
    df_PDBe_PDB_UniProt, df_PDBe_PDB_UniProt_WOnull = make_df_from_SIFTS_data(PDBe_PDB, combined_PDBe_UniProt_AccessionID, 
                                                                              default_num, chains_to_change)

    chain_total_renum, nothing_changed = count_renumbered_in_chains(chains_to_change_1toN, df_PDBe_PDB_UniProt_WOnull, mmCIF_name, 
                                                                    UniProt_conversion_dict, longest_AccessionIDs, default_num)
    
    chain_total_renum.append(nothing_changed)
    mod_log_message = chain_total_renum

    # for no change needed _no_change_out.cif.gz
    if nothing_changed == True:
        copy_file(default_input_path_to_mmCIF, mmCIF_name, default_output_path_to_mmCIF, "_nochange.cif.gz", gzip_mode)
        return mod_log_message

    df_final_atom_site, mmcif_dict = mmCIF_parser(mmCIF_name, default_input_path_to_mmCIF, df_PDBe_PDB_UniProt_WOnull,
                                                  default_num, chains_to_change, chains_to_change_1toN)
    
    poly_nonpoly_concat = poly_nonpoly_renum(mmcif_dict, df_PDBe_PDB_UniProt, chains_to_change, default_num)
    poly_nonpoly_atom_site = pd.concat([poly_nonpoly_concat, df_final_atom_site], ignore_index=True).drop_duplicates(subset="PDB_num_and_chain", keep='first', sort=True)

    formed_columns = column_formation(mmcif_dict)
    mmcif_dict = renumber_tables(formed_columns, mmcif_dict, poly_nonpoly_atom_site, chains_to_change, default_num)

    try:
        output_with_this_name_ending("_renum.cif", default_output_path_to_mmCIF, mmcif_dict, mmCIF_name=mmCIF_name, gzip_mode=gzip_mode)
        return mod_log_message
    except IndexError:
        print("IndexError Warning this file is not renumbered:", mmCIF_name)
        copy_file(default_input_path_to_mmCIF, mmCIF_name, default_output_path_to_mmCIF, ".cif.gz", gzip_mode)
        
        
### Main loop
with ProcessPoolExecutor() as executor:
    # Start the processes
    results = []
    futures = [executor.submit(renum_mmCIF, pdbfn) for pdbfn in active_pdbfn]

    # Collect the results as they become available
    for future in tqdm(as_completed(futures), total=len(futures)):
        pass
