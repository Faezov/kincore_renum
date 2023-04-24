import os
import re
import shutil
import gzip
from pathlib import Path

import Bio
from Bio.PDB import *
import xml.etree.ElementTree as ET  # xml parser
import pandas as pd
import numpy as np


from src.download_data import url_formation_for_pool, download_with_pool
exception_AccessionIDs = ["P42212", "Q17104", "Q27903", "Q93125", "P03069", 
                          "D3DLN9", "Q96UT3", "P0ABE7", "P00192", "P76805", 
                          "Q8XCE3", "P00720", "Q38170", "Q94N07", "P0AEX9", 
                          "P02928", "Q2M6S0"]

def try_mmCIF2dict(default_input_path_to_mmcif, mmcif_name, max_retries=3):
    mmcif_dict = None
    mmcif_path = Path(default_input_path_to_mmcif).joinpath(mmcif_name)
    for _ in range(max_retries):
        try:
            with gzip.open(mmcif_path, 'rt') as mmcif_file:
                mmcif_dict = Bio.PDB.MMCIF2Dict.MMCIF2Dict(mmcif_file)
            break
        except (EOFError, ValueError, OSError):
            if mmcif_path.exists():
                os.remove(mmcif_path)
            if "assembly" in mmcif_name:
                download_with_pool(url_formation_for_pool("mmCIF_assembly", [mmcif_name]))
            else:
                download_with_pool(url_formation_for_pool("mmCIF", [mmcif_name]))
                
    return mmcif_dict

def try_SIFTS_tree_parser(default_input_path_to_SIFTS, SIFTS_name, max_retries=3):
    SIFTS_tree_parser_product = None
    SIFTS_path = Path(default_input_path_to_SIFTS).joinpath(SIFTS_name)

    for _ in range(max_retries):
        try:
            with gzip.open(SIFTS_path, 'rt') as handle_SIFTS:
                SIFTS_tree_parser_product = SIFTS_tree_parser(handle_SIFTS)
            break
        except (EOFError, ValueError, OSError):
            if SIFTS_path.exists():
                os.remove(SIFTS_path)
            download_with_pool(url_formation_for_pool("SIFTS", [SIFTS_name]))

    return SIFTS_tree_parser_product


def compress_output_files(full_path_to_the_file, gzip_mode="on"):
    if gzip_mode == "on":
        input_path = Path(full_path_to_the_file)
        output_path = input_path.with_suffix(input_path.suffix + '.gz')
        
        with input_path.open('rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def output_with_this_name_ending(suffix, path, mmcif_dict, mmCIF_name, gzip_mode="on"):
    mmCIF_name = mmCIF_name.replace(".gz", "").replace(".cif", "")
    output_path = Path(path)
    output_file = output_path / (mmCIF_name + suffix)
    
    io = MMCIFIO()
    io.set_dict(mmcif_dict)

    with open(output_file, 'w') as outfile:
        io.save(outfile)
    
    if gzip_mode == "on":
        compress_output_files(output_file, gzip_mode)
        output_file.unlink()
        
def copy_file(inpath, file_name, outpath, suffix, gzip_mode):
    mmCIF_name = file_name[:file_name.rfind(".cif.gz")]
    absolute_path_in = Path(inpath) / file_name
    absolute_path_out = Path(outpath) / (mmCIF_name + suffix)
    
    if gzip_mode == "off":
        with gzip.open(absolute_path_in, 'rb') as f_in:
            with absolute_path_out.with_suffix('').open('wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        shutil.copyfile(absolute_path_in, absolute_path_out)


def SIFTS_tree_parser(handle_SIFTS):
    tree = ET.parse(handle_SIFTS)
    root = tree.getroot()
    
    # tuples resnum, resname, chainid
    PDBe = list()
    PDBe_for_Uni = list()
    PDBe_for_PDB = list()
    PDB = list()
    UniProt = list()
    
    AccessionID = list()
    UniProt_conversion_dict = dict()

    for entity in root:
        if entity.tag.endswith("entity"):
            entity_chainID_list = list(entity.attrib.items())
            if entity_chainID_list[0][0] == "type" and entity_chainID_list[0][1] == "protein":
                for segment in entity:
                    for listResidue in segment:
                        if listResidue.tag.endswith("listMapRegion"):
                            for mapRegion in listResidue:
                                for db in mapRegion:
                                    dbSource_UniProt = list(db.attrib.items())
                                    if "dbSource" == dbSource_UniProt[0][0] and "UniProt" == dbSource_UniProt[0][1]:
                                        if db.text is None:
                                            UniProt_code = dbSource_UniProt[2][1]
                                        else:
                                            Human_readable = db.text
                                            UniProt_conversion_dict[UniProt_code] = Human_readable

                        for residue in listResidue:
                            parent = list(residue.attrib.items())
                            if parent[0][0] == "dbSource" and parent[0][1] == "PDBe":
                                PDBe.append((parent[2][1], parent[3][1], entity_chainID_list[1][1]))

                                for crossRefDb in residue:
                                    child = list(crossRefDb.attrib.items())
                                    if child[0][0] == "dbSource" and child[0][1] == "PDB":
                                        PDB.append((child[3][1], child[4][1], child[5][1]))
                                        PDBe_for_PDB.append((parent[2][1], parent[3][1], entity_chainID_list[1][1]))

                                    if child[0][0] == "dbSource" and child[0][1] == "UniProt":
                                        UniProt.append((child[3][1], child[4][1], entity_chainID_list[1][1]))
                                        PDBe_for_Uni.append((parent[2][1], parent[3][1], entity_chainID_list[1][1]))
                                        AccessionID.append(child[2][1])
    ### resname,resnum, chainid
    PDBe_PDB = list(zip(PDBe_for_PDB, PDB))
    PDBe_UniProt_AccessionID = list(zip(PDBe_for_Uni, UniProt, AccessionID))

    return [PDBe_PDB, PDBe_UniProt_AccessionID, UniProt_conversion_dict]
    
    
def make_df_from_SIFTS_data(PDBe_PDB, PDBe_UniProt_AccessionID, default_num=50000, chains_to_change="all"):
    df_PDBe_UniProt = pd.DataFrame(PDBe_UniProt_AccessionID, columns=['PDBe', 'UniProt', "AccessionID"]).drop_duplicates(subset="PDBe", keep='first')
    df_PDBe_PDB = pd.DataFrame(PDBe_PDB, columns=['PDBe', 'PDB']).drop_duplicates(subset="PDBe", keep='first')

    df_PDBe_PDB_UniProt = df_PDBe_PDB.merge(df_PDBe_UniProt, on="PDBe", how='left')
    df_PDBe_PDB_UniProt['UniProt'] = df_PDBe_PDB_UniProt['UniProt'].replace(np.nan, str(default_num))
    df_PDBe_PDB_UniProt["Uni_moD"] = np.where(df_PDBe_PDB_UniProt['UniProt'] != str(default_num), df_PDBe_PDB_UniProt['UniProt'], df_PDBe_PDB_UniProt["PDBe"])
    df_PDBe_PDB_UniProt['new_col_Uni'] = df_PDBe_PDB_UniProt['Uni_moD'].apply(lambda x: x[0])
    df_PDBe_PDB_UniProt['UniProt_50k'] = df_PDBe_PDB_UniProt['new_col_Uni'].apply(lambda x: str(int(x) + default_num) if type(x) == str else x)
    df_PDBe_PDB_UniProt.loc[df_PDBe_PDB_UniProt['UniProt'] != str(default_num), 'UniProt_50k'] = df_PDBe_PDB_UniProt['new_col_Uni']

    PDBe_newnum_UniProt_PDB_AccessionID = []

    for index, rows in df_PDBe_PDB_UniProt.iterrows():
        if chains_to_change == "all" or rows.PDB[2].strip() in chains_to_change:
            intermediate_list = [rows.PDBe, rows.UniProt_50k, rows.Uni_moD, rows.PDB, rows.AccessionID]
        else:
            intermediate_list = [rows.PDBe, rows.PDB[0], rows.Uni_moD, rows.PDB, rows.AccessionID]
        PDBe_newnum_UniProt_PDB_AccessionID.append(intermediate_list)

    df_PDBe_PDB_UniProt["PDBe_newnum_UniProt_PDB_AccessionID"] = PDBe_newnum_UniProt_PDB_AccessionID
    df_PDBe_PDB_UniProt_WOnull = df_PDBe_PDB_UniProt[df_PDBe_PDB_UniProt.PDB.map(lambda x: x[0]) != "null"].set_index("PDBe")

    return [df_PDBe_PDB_UniProt, df_PDBe_PDB_UniProt_WOnull]
    
def get_chains_and_accessions(df_PDBe_PDB_UniProt):
    chains_to_change = set()
    chains_to_change_1toN = set()
    AccessionIDs = set()
    chain_AccessionID_dict = dict()

    for item in df_PDBe_PDB_UniProt["PDBe_newnum_UniProt_PDB_AccessionID"]:
        if type(item[4]) == float:
            continue
        chains_to_change.add(item[3][2])
        chains_to_change_1toN.add(item[2][2])
        AccessionIDs.add(item[4])

    for chain in chains_to_change:
        AccessionIDs_in_chain = set()
        for item in df_PDBe_PDB_UniProt["PDBe_newnum_UniProt_PDB_AccessionID"]:
            if chain == item[3][2]:
                if item[4] is not np.nan:
                    AccessionIDs_in_chain.add(item[4])
        chain_AccessionID_dict[chain] = AccessionIDs_in_chain

    return chains_to_change, chains_to_change_1toN, AccessionIDs, chain_AccessionID_dict


def resolve_numbering_clashes(df_PDBe_PDB_UniProt, exception_AccessionIDs, chain_AccessionID_dict):
    combined_PDBe_UniProt_AccessionID = list()
    longest_AccessionIDs = list()
    clash = 0

    for chain, accessions in chain_AccessionID_dict.items():
        longest_AccessionID = None
        longest_PDBe_UniProt_AccessionID = list()

        if len(accessions) > 1:
            for accession in accessions:
                PDBe_UniProt_AccessionID = list()
                target_UniProt_numbers_in_chain = list()
                diff_UniProt_numbers_in_same_chain = list()
                diff_PDBe_UniProt_AccessionID = list()

                for item in df_PDBe_PDB_UniProt["PDBe_newnum_UniProt_PDB_AccessionID"]:
                    if (item[4] == accession and item[3][2] == chain and item[4] is not np.nan):
                        PDBe_UniProt_AccessionID.append((item[0], item[2], item[4]))
                        target_UniProt_numbers_in_chain.append(item[2])

                    if (item[4] != accession and item[3][2] == chain and item[4] is not np.nan):
                        diff_UniProt_numbers_in_same_chain.append(item[2])
                        diff_PDBe_UniProt_AccessionID.append((item[0], item[2], item[4]))

                for target_Uni in target_UniProt_numbers_in_chain:
                    for diff_Uni in diff_UniProt_numbers_in_same_chain:
                        if target_Uni[0] == diff_Uni[0]:
                            clash = 1

                if accession not in exception_AccessionIDs or longest_AccessionID is None:
                    if len(longest_PDBe_UniProt_AccessionID) < len(PDBe_UniProt_AccessionID):
                        longest_PDBe_UniProt_AccessionID = PDBe_UniProt_AccessionID
                        longest_AccessionID = accession

            if clash == 1:
                combined_PDBe_UniProt_AccessionID.extend(longest_PDBe_UniProt_AccessionID)
                longest_AccessionIDs.append(longest_AccessionID)
            else:
                combined_PDBe_UniProt_AccessionID.extend(longest_PDBe_UniProt_AccessionID)
                combined_PDBe_UniProt_AccessionID.extend(diff_PDBe_UniProt_AccessionID)
        else:
            for accession in accessions:
                PDBe_UniProt_AccessionID = list()
                target_UniProt_numbers_in_chain = list()

                for item in df_PDBe_PDB_UniProt["PDBe_newnum_UniProt_PDB_AccessionID"]:
                    if (item[4] == accession and item[3][2] == chain and item[4] is not np.nan):
                        PDBe_UniProt_AccessionID.append((item[0], item[2], item[4]))
                        target_UniProt_numbers_in_chain.append(item[2])

            combined_PDBe_UniProt_AccessionID.extend(PDBe_UniProt_AccessionID)

    return [combined_PDBe_UniProt_AccessionID, longest_AccessionIDs]
    
    
def count_renumbered_in_chains(chains_to_change_1toN, df_PDBe_PDB_UniProt_WOnull, mmCIF_name,
                               UniProt_conversion_dict, longest_AccessionIDs, default_num):
    nothing_changed = False
    chain_total_renum = list()
    chain_PDBe_PDB = dict()
    
    count_renum_for_all_chains = 0
    count_default_num_for_all_chains = 0
    

    for chain in sorted(chains_to_change_1toN):
        total_count_per_chain = 0
        count_default_num = 0
        UniProt_set = set()

        for PDBe_num_Uni_PDB in df_PDBe_PDB_UniProt_WOnull["PDBe_newnum_UniProt_PDB_AccessionID"]:
            if chain == PDBe_num_Uni_PDB[2][2]:
                chain_PDBe_PDB[chain] = PDBe_num_Uni_PDB[3][2]
                if type(PDBe_num_Uni_PDB[4]) != float:
                    UniProt_set.add(PDBe_num_Uni_PDB[4])
                total_count_per_chain += 1
                if int(PDBe_num_Uni_PDB[1]) > int(default_num):
                    count_default_num += 1
                    count_default_num_for_all_chains += 1
                elif PDBe_num_Uni_PDB[1] != PDBe_num_Uni_PDB[3][0]:
                    count_renum_for_all_chains += 1


        for accession in UniProt_set:
            renum_for_accession = 0
            count_accession_len = 0
            for PDBe_num_Uni_PDB in df_PDBe_PDB_UniProt_WOnull["PDBe_newnum_UniProt_PDB_AccessionID"]:
                if accession == PDBe_num_Uni_PDB[4]:
                    if chain == PDBe_num_Uni_PDB[2][2]:
                        count_accession_len += 1
                if chain == PDBe_num_Uni_PDB[2][2] and accession == PDBe_num_Uni_PDB[4]:
                    if PDBe_num_Uni_PDB[1] != PDBe_num_Uni_PDB[3][0]:
                        renum_for_accession += 1

            if len(longest_AccessionIDs) != 0:
                if accession in longest_AccessionIDs:
                    accessionID_readable_longest = UniProt_conversion_dict.get(accession)
                    chain_total_renum.append(
                        [mmCIF_name[:4] + "*", chain, chain_PDBe_PDB[chain], accession, 
                         accessionID_readable_longest, count_accession_len,
                         total_count_per_chain, renum_for_accession, count_default_num])
                else:
                    accessionID_readable = UniProt_conversion_dict.get(accession)
                    chain_total_renum.append(
                        [mmCIF_name[:4], chain, chain_PDBe_PDB[chain], accession, 
                         accessionID_readable, count_accession_len, 
                         total_count_per_chain, renum_for_accession, count_default_num])
            else:
                if type(accession) != float:
                    accessionID_readable = UniProt_conversion_dict.get(accession)
                    chain_total_renum.append(
                        [mmCIF_name[:4], chain, chain_PDBe_PDB[chain], accession, 
                         accessionID_readable, count_accession_len, 
                         total_count_per_chain, renum_for_accession, count_default_num])

    if count_renum_for_all_chains == 0 and count_default_num_for_all_chains == 0:
        nothing_changed = True

    return [chain_total_renum, nothing_changed]
    
    
def if_no_SIFTS_data_log(mmCIF_name, mmcif_dict, log_message):
    keys_to_try = [["_pdbx_poly_seq_scheme.asym_id", "_pdbx_poly_seq_scheme.pdb_strand_id"],
                   ["_pdbe_orig_poly_seq_scheme.asym_id", "_pdbe_orig_poly_seq_scheme.pdb_strand_id"],
                   ["_atom_site.label_asym_id", "_atom_site.auth_asym_id"]]

    for key_pair in keys_to_try:
        try:
            pull_1toN_chainID = mmcif_dict[key_pair[0]]
            pull_auth_chainID = mmcif_dict[key_pair[1]]
            break
        except KeyError:
            continue
    else:
        raise KeyError("No suitable keys found in the mmcif_dict")

    df = pd.DataFrame({'pull_1toN_chainID': pull_1toN_chainID, 'pull_auth_chainID': pull_auth_chainID}).drop_duplicates()

    for label_asym_id in sorted(set(pull_1toN_chainID)):
        ### mmCIF_name, 1toN_chain, auth_chain, accession, accessionID_readable, 
        ### count_accession_len, total_count_per_chain, renum_for_accession, count_default_num
        
        count_elements_in_strand = sum(1 for chain_id in pull_1toN_chainID if chain_id == label_asym_id)
        auth_asym_id = df.loc[df['pull_1toN_chainID'] == label_asym_id, 'pull_auth_chainID'].values[0]
        log_message.append([mmCIF_name[:4], label_asym_id, auth_asym_id, "-", "-", "-", count_elements_in_strand, "0", "0"])

    return log_message

def mmCIF_parser(mmCIF_name, default_input_path_to_mmCIF, df_PDBe_PDB_UniProt_WOnull, default_num, chains_to_change, chains_to_change_1toN):
    mmcif_dict = try_mmCIF2dict(default_input_path_to_mmCIF, mmCIF_name)
    if mmcif_dict == None:
        return None
    
    try:
        _pdbx_poly_seq_scheme_auth_seq_num_before_change = mmcif_dict["_pdbx_poly_seq_scheme.auth_seq_num"]
    except KeyError:
        _pdbx_poly_seq_scheme_auth_seq_num_before_change = mmcif_dict["_pdbe_orig_poly_seq_scheme.auth_seq_num"]
        pass

    _atom_site_label_comp_id = mmcif_dict["_atom_site.label_comp_id"]
    _atom_site_label_seq_id = mmcif_dict["_atom_site.label_seq_id"]
    _atom_site_label_asym_id = mmcif_dict["_atom_site.label_asym_id"]
    _atom_site_pdbx_PDB_ins_code = mmcif_dict["_atom_site.pdbx_PDB_ins_code"]

    _atom_site_auth_comp_id = mmcif_dict["_atom_site.auth_comp_id"]
    _atom_site_auth_seq_id = mmcif_dict["_atom_site.auth_seq_id"]
    _atom_site_auth_asym_id = mmcif_dict["_atom_site.auth_asym_id"]
    _atom_site_pdbx_formal_charge = mmcif_dict["_atom_site.pdbx_formal_charge"]

    mmCIF_label = list(zip(_atom_site_label_seq_id, _atom_site_label_comp_id, _atom_site_label_asym_id))
    mmCIF_auth = list(zip(_atom_site_auth_seq_id, _atom_site_auth_comp_id, _atom_site_auth_asym_id))
    mmCIF_label_auth_inscode = list(zip(mmCIF_label, mmCIF_auth, _atom_site_pdbx_PDB_ins_code))

    df_mmCIF = pd.DataFrame(mmCIF_label_auth_inscode)
    df_mmCIF = df_mmCIF.rename(columns={0: "OnetoN_mmCIF", 1: "auth_mmCIF", 2: "ins_code"})

    df_mmCIF["PDBnum_inc_code"] = np.where(df_mmCIF['ins_code'] != "?",
                                           (df_mmCIF['auth_mmCIF'].apply(lambda x: x[0]) + df_mmCIF["ins_code"].apply(lambda y: y[0]) + ","
                                            + df_mmCIF['auth_mmCIF'].apply(lambda x: x[1]) + "," + df_mmCIF['auth_mmCIF'].apply(lambda x: x[2])),
                                           df_mmCIF["ins_code"])
    df_mmCIF["PDBnum_inc_code_cor"] = np.where(df_mmCIF["PDBnum_inc_code"] != "?", df_mmCIF["PDBnum_inc_code"].apply(lambda x: tuple(x.split(","))),
                                               df_mmCIF["auth_mmCIF"])

    df_mmCIF["auth_mmCIF"] = df_mmCIF["PDBnum_inc_code_cor"]
    df_mmCIF = df_mmCIF.drop(columns=["PDBnum_inc_code_cor", "ins_code", "PDBnum_inc_code"])

    df_PDBe_PDB_UniProt_WOnull = df_PDBe_PDB_UniProt_WOnull.reset_index()
    df_final = df_mmCIF.merge(df_PDBe_PDB_UniProt_WOnull, left_on="OnetoN_mmCIF", right_on="PDBe", how='left')    
    df_final = df_final.rename(columns={"PDBe_copy": "PDBe"})
    df_final = df_final.drop_duplicates(subset="auth_mmCIF", keep='first')
    df_final["PDB_num_and_chain"] = df_final["auth_mmCIF"].apply(lambda x: (x[0], x[2]))
    df_final["PDBe_num_and_chain"] = df_final["OnetoN_mmCIF"].apply(lambda x: (x[0], x[2]))

    df_final["Uni_or_50k_NAN"] = np.where(df_final["OnetoN_mmCIF"].apply(lambda x: x[0] != "."),
                                          df_final["UniProt_50k"].apply(lambda x: x),
                                          df_final["PDB_num_and_chain"].apply(
                                              lambda x: str(int(''.join(filter(str.isdigit, x[0]))) + default_num + 10000)
                                              if x[1] in chains_to_change else str(int(''.join(filter(str.isdigit, x[0]))))))
    df_final["Uni_or_50k"] = np.where(df_final["Uni_or_50k_NAN"].apply(lambda x: type(x) == float),
                                      df_final["PDBe_num_and_chain"].apply(
                                          lambda x: "." if x[0] == "." else str(int(''.join(filter(str.isdigit, x[0]))) + default_num)
                                          if x[1] in chains_to_change_1toN else str(int(''.join(filter(str.isdigit, x[0]))))),
                                      df_final["Uni_or_50k_NAN"].apply(lambda x: x))

    df_final_atom_site = df_final[["PDBe", "PDB", "UniProt", "PDBe_num_and_chain", "PDB_num_and_chain", "AccessionID", "Uni_or_50k"]]

    return [df_final_atom_site, mmcif_dict]
    
def poly_nonpoly_renum(mmcif_dict, df_PDBe_PDB_UniProt, chains_to_change, default_num):
    try:
        _pdbx_poly_seq_scheme_seq_id = mmcif_dict["_pdbx_poly_seq_scheme.seq_id"]
        _pdbx_poly_seq_scheme_asym_id = mmcif_dict["_pdbx_poly_seq_scheme.asym_id"]
        _pdbx_poly_seq_scheme_mon_id = mmcif_dict["_pdbx_poly_seq_scheme.mon_id"]

        _pdbx_poly_seq_scheme_pdb_seq_num = mmcif_dict["_pdbx_poly_seq_scheme.pdb_seq_num"]
        _pdbx_poly_seq_scheme_auth_seq_num = mmcif_dict["_pdbx_poly_seq_scheme.auth_seq_num"]
        _pdbx_poly_seq_scheme_pdb_mon_id = mmcif_dict["_pdbx_poly_seq_scheme.pdb_mon_id"]
        _pdbx_poly_seq_scheme_auth_mon_id = mmcif_dict["_pdbx_poly_seq_scheme.auth_mon_id"]
        _pdbx_poly_seq_scheme_pdb_strand_id = mmcif_dict["_pdbx_poly_seq_scheme.pdb_strand_id"]
        _pdbx_poly_seq_scheme_pdb_ins_code = mmcif_dict["_pdbx_poly_seq_scheme.pdb_ins_code"]
    except KeyError:
        try:
            _pdbx_poly_seq_scheme_seq_id = mmcif_dict["_pdbe_orig_poly_seq_scheme.seq_id"]
            _pdbx_poly_seq_scheme_asym_id = mmcif_dict["_pdbe_orig_poly_seq_scheme.asym_id"]
            _pdbx_poly_seq_scheme_mon_id = mmcif_dict["_pdbe_orig_poly_seq_scheme.mon_id"]

            _pdbx_poly_seq_scheme_pdb_seq_num = mmcif_dict["_pdbe_orig_poly_seq_scheme.pdb_seq_num"]
            _pdbx_poly_seq_scheme_auth_seq_num = mmcif_dict["_pdbe_orig_poly_seq_scheme.auth_seq_num"]
            _pdbx_poly_seq_scheme_pdb_mon_id = mmcif_dict["_pdbe_orig_poly_seq_scheme.pdb_mon_id"]
            _pdbx_poly_seq_scheme_auth_mon_id = mmcif_dict["_pdbe_orig_poly_seq_scheme.auth_mon_id"]
            _pdbx_poly_seq_scheme_pdb_strand_id = mmcif_dict["_pdbe_orig_poly_seq_scheme.pdb_strand_id"]
            _pdbx_poly_seq_scheme_pdb_ins_code = mmcif_dict["_pdbe_orig_poly_seq_scheme.pdb_ins_code"]

        except KeyError:
            return 0

    if type(_pdbx_poly_seq_scheme_pdb_strand_id) == str:
        _pdbx_poly_seq_scheme_pdb_seq_num = [_pdbx_poly_seq_scheme_pdb_seq_num]
        _pdbx_poly_seq_scheme_auth_seq_num = [_pdbx_poly_seq_scheme_auth_seq_num]
        _pdbx_poly_seq_scheme_pdb_mon_id = [_pdbx_poly_seq_scheme_pdb_mon_id]
        _pdbx_poly_seq_scheme_auth_mon_id = [_pdbx_poly_seq_scheme_auth_mon_id]
        _pdbx_poly_seq_scheme_pdb_strand_id = [_pdbx_poly_seq_scheme_pdb_strand_id]
        _pdbx_poly_seq_scheme_pdb_ins_code = [_pdbx_poly_seq_scheme_pdb_ins_code]

    mmCIF_pdbx_poly_seq_scheme_label = list(zip(_pdbx_poly_seq_scheme_seq_id,
                                                _pdbx_poly_seq_scheme_mon_id,
                                                _pdbx_poly_seq_scheme_asym_id))
    mmCIF_pdbx_poly_seq_scheme_pdb = list(zip(_pdbx_poly_seq_scheme_pdb_seq_num,
                                              _pdbx_poly_seq_scheme_pdb_mon_id,
                                              _pdbx_poly_seq_scheme_pdb_strand_id))
    mmCIF_pdbx_poly_seq_scheme_auth = list(zip(_pdbx_poly_seq_scheme_auth_seq_num,
                                               _pdbx_poly_seq_scheme_auth_mon_id,
                                               _pdbx_poly_seq_scheme_pdb_strand_id))

    df_mmCIF_pdbx_poly_seq_scheme = pd.DataFrame(zip(mmCIF_pdbx_poly_seq_scheme_label,
                                                     mmCIF_pdbx_poly_seq_scheme_pdb,
                                                     mmCIF_pdbx_poly_seq_scheme_auth,
                                                     _pdbx_poly_seq_scheme_pdb_ins_code))

    df_mmCIF_pdbx_poly_seq_scheme = df_mmCIF_pdbx_poly_seq_scheme.rename(
        columns={0: "_pdbx_poly_seq_scheme_label", 1: "_pdbx_poly_seq_scheme_pdb",
                 2: "_pdbx_poly_seq_scheme_auth", 3: "_pdbx_poly_seq_scheme_pdb_ins_code"})

    df_pdbx_poly_seq_scheme_pdb_final = df_mmCIF_pdbx_poly_seq_scheme.merge(
        df_PDBe_PDB_UniProt, left_on="_pdbx_poly_seq_scheme_label", right_on="PDBe", how='left')
    df_pdbx_poly_seq_scheme_pdb_final["PDBe_num_and_chain"] = df_pdbx_poly_seq_scheme_pdb_final[
        "_pdbx_poly_seq_scheme_label"].apply(lambda x: (x[0], x[2]))

    df_pdbx_poly_seq_scheme_pdb_final["PDB_num_and_chain"] = np.where(
        df_pdbx_poly_seq_scheme_pdb_final["_pdbx_poly_seq_scheme_pdb_ins_code"].apply(lambda x: x == "."),
        df_pdbx_poly_seq_scheme_pdb_final["_pdbx_poly_seq_scheme_pdb"].apply(lambda x: (x[0], x[2])),
        df_pdbx_poly_seq_scheme_pdb_final["_pdbx_poly_seq_scheme_pdb"].apply(lambda x: x[0]) +
        df_pdbx_poly_seq_scheme_pdb_final["_pdbx_poly_seq_scheme_pdb_ins_code"].apply(lambda x: x) + "," +
        df_pdbx_poly_seq_scheme_pdb_final["_pdbx_poly_seq_scheme_pdb"].apply(lambda x: x[2]))
    df_pdbx_poly_seq_scheme_pdb_final["PDB_num_and_chain"] = df_pdbx_poly_seq_scheme_pdb_final["PDB_num_and_chain"].apply(
        lambda x: tuple(x.split(",")) if type(x) == str else x)

    df_pdbx_poly_seq_scheme_pdb_final["Uni_or_50k"] = np.where(
        df_pdbx_poly_seq_scheme_pdb_final["PDB_num_and_chain"].apply(lambda x: x[1] in chains_to_change),
        df_pdbx_poly_seq_scheme_pdb_final["UniProt_50k"].apply(lambda x: x),
        df_pdbx_poly_seq_scheme_pdb_final["PDB_num_and_chain"].apply(lambda x: x[0].strip(re.sub('[0-9\-\?\.]+', '', x[0]))))

    try:
        mmcif_dict["_pdbx_poly_seq_scheme.pdb_seq_num"]  # check if key exists
        mmcif_dict["_pdbx_poly_seq_scheme.pdb_seq_num"] = list(df_pdbx_poly_seq_scheme_pdb_final["Uni_or_50k"].values)
        mmcif_dict["_pdbx_poly_seq_scheme.auth_seq_num"] = _pdbx_poly_seq_scheme_pdb_seq_num
    except KeyError:
        mmcif_dict["_pdbe_orig_poly_seq_scheme.pdb_seq_num"] = list(df_pdbx_poly_seq_scheme_pdb_final["Uni_or_50k"].values)
        mmcif_dict["_pdbe_orig_poly_seq_scheme.auth_seq_num"] = _pdbx_poly_seq_scheme_pdb_seq_num

    nonpoly_present = False

    try:
        _pdbx_nonpoly_scheme_pdb_seq_num = mmcif_dict["_pdbx_nonpoly_scheme.pdb_seq_num"]
        _pdbx_nonpoly_scheme_auth_seq_num = mmcif_dict["_pdbx_nonpoly_scheme.auth_seq_num"]
        _pdbx_nonpoly_scheme_pdb_mon_id = mmcif_dict["_pdbx_nonpoly_scheme.pdb_mon_id"]
        _pdbx_nonpoly_scheme_auth_mon_id = mmcif_dict["_pdbx_nonpoly_scheme.auth_mon_id"]
        _pdbx_nonpoly_scheme_pdb_strand_id = mmcif_dict["_pdbx_nonpoly_scheme.pdb_strand_id"]
        _pdbx_nonpoly_scheme_asym_id = mmcif_dict["_pdbx_nonpoly_scheme.asym_id"]
        dots_for_label = ["." for _ in range(len(_pdbx_nonpoly_scheme_asym_id)) if type(_pdbx_nonpoly_scheme_asym_id) == list]
        nonpoly_present = True
    except KeyError:
        try:
            _pdbx_nonpoly_scheme_pdb_seq_num = mmcif_dict["_pdbe_orig_nonpoly_scheme.pdb_seq_num"]
            _pdbx_nonpoly_scheme_auth_seq_num = mmcif_dict["_pdbe_orig_nonpoly_scheme.auth_seq_num"]
            _pdbx_nonpoly_scheme_pdb_mon_id = mmcif_dict["_pdbe_orig_nonpoly_scheme.pdb_mon_id"]
            _pdbx_nonpoly_scheme_auth_mon_id = mmcif_dict["_pdbe_orig_nonpoly_scheme.auth_mon_id"]
            _pdbx_nonpoly_scheme_pdb_strand_id = mmcif_dict["_pdbe_orig_nonpoly_scheme.pdb_strand_id"]
            _pdbx_nonpoly_scheme_asym_id = mmcif_dict["_pdbe_orig_nonpoly_scheme.asym_id"]
            dots_for_label = ["." for _ in range(len(_pdbx_nonpoly_scheme_asym_id)) if type(_pdbx_nonpoly_scheme_asym_id) == list]
            nonpoly_present = True
        except KeyError:
            pass

    if nonpoly_present:
        if type(_pdbx_nonpoly_scheme_pdb_strand_id) == str:
            _pdbx_nonpoly_scheme_pdb_seq_num = [_pdbx_nonpoly_scheme_pdb_seq_num]
            _pdbx_nonpoly_scheme_auth_seq_num = [_pdbx_nonpoly_scheme_auth_seq_num]
            _pdbx_nonpoly_scheme_pdb_mon_id = [_pdbx_nonpoly_scheme_pdb_mon_id]
            _pdbx_nonpoly_scheme_auth_mon_id = [_pdbx_nonpoly_scheme_auth_mon_id]
            _pdbx_nonpoly_scheme_pdb_strand_id = [_pdbx_nonpoly_scheme_pdb_strand_id]
            _pdbx_nonpoly_scheme_asym_id = [_pdbx_nonpoly_scheme_asym_id]
            dots_for_label = ["."]

        mmCIF_pdbx_nonpoly_scheme_pdb = list(zip(_pdbx_nonpoly_scheme_pdb_seq_num,
                                                 _pdbx_nonpoly_scheme_pdb_mon_id,
                                                 _pdbx_nonpoly_scheme_pdb_strand_id))
        mmCIF_pdbx_nonpoly_scheme_auth = list(zip(_pdbx_nonpoly_scheme_auth_seq_num,
                                                  _pdbx_nonpoly_scheme_auth_mon_id,
                                                  _pdbx_nonpoly_scheme_pdb_strand_id))
        mmCIF_pdbx_nonpoly_scheme_label = list(zip(dots_for_label,
                                                   _pdbx_nonpoly_scheme_pdb_mon_id,
                                                   _pdbx_nonpoly_scheme_asym_id))

        df_mmCIF_pdbx_nonpoly_scheme = pd.DataFrame(zip(mmCIF_pdbx_nonpoly_scheme_pdb,
                                                        mmCIF_pdbx_nonpoly_scheme_auth,
                                                        mmCIF_pdbx_nonpoly_scheme_label))
        df_mmCIF_pdbx_nonpoly_scheme = df_mmCIF_pdbx_nonpoly_scheme.rename(columns={0: "pdbx_nonpoly_scheme_pdb",
                                                                                    1: "pdbx_nonpoly_scheme_auth",
                                                                                    2: "pdbx_nonpoly_scheme_label"})

        df_mmCIF_pdbx_nonpoly_scheme["PDB"] = df_mmCIF_pdbx_nonpoly_scheme["pdbx_nonpoly_scheme_pdb"]
        df_mmCIF_pdbx_nonpoly_scheme["PDB_num_and_chain"] = df_mmCIF_pdbx_nonpoly_scheme["pdbx_nonpoly_scheme_pdb"].apply(lambda x: (x[0], x[2]))
        df_mmCIF_pdbx_nonpoly_scheme["Uni_or_50k"] = df_mmCIF_pdbx_nonpoly_scheme["pdbx_nonpoly_scheme_pdb"].apply(
            lambda x: str(int(x[0]) + default_num + 10000) if x[2] in chains_to_change else x[0])

        try:
            mmcif_dict["_pdbx_nonpoly_scheme.pdb_seq_num"]  # check if key exists
            mmcif_dict["_pdbx_nonpoly_scheme.pdb_seq_num"] = list(df_mmCIF_pdbx_nonpoly_scheme["Uni_or_50k"].values)
            mmcif_dict["_pdbx_nonpoly_scheme.auth_seq_num"] = _pdbx_nonpoly_scheme_pdb_seq_num
        except KeyError:
            try:
                mmcif_dict["_pdbe_orig_nonpoly_scheme.pdb_seq_num"] = list(df_mmCIF_pdbx_nonpoly_scheme["Uni_or_50k"].values)
                mmcif_dict["_pdbe_orig_nonpoly_scheme.auth_seq_num"] = _pdbx_nonpoly_scheme_pdb_seq_num
            except KeyError:
                pass
            
        poly_nonpoly_concat = pd.concat([df_pdbx_poly_seq_scheme_pdb_final, df_mmCIF_pdbx_nonpoly_scheme], ignore_index=True)
        poly_nonpoly_concat = poly_nonpoly_concat[["PDBe", "PDB", "UniProt", "PDBe_num_and_chain", "PDB_num_and_chain", "AccessionID", "Uni_or_50k"]]
    else:
        poly_nonpoly_concat = df_pdbx_poly_seq_scheme_pdb_final[["PDBe", "PDB", "UniProt", "PDBe_num_and_chain", "PDB_num_and_chain", "AccessionID", "Uni_or_50k"]]

    return poly_nonpoly_concat
    
def column_formation(mmcif_dict):
    mmcif_dict_keys = mmcif_dict.keys()
    aut_seq_all_splitted = list()
    for key in mmcif_dict_keys:
        key_dot_splitted = key.split(".")
        for tab_name_col_name in key_dot_splitted:
            if "auth_seq" in tab_name_col_name:
                if "auth_seq_id" in key:
                    aut_seq_all_splitted.append(key_dot_splitted[:1] + key_dot_splitted[1].split("auth_seq_id"))
                if "auth_seq_num" in key:
                    aut_seq_all_splitted.append(key_dot_splitted[:1] + key_dot_splitted[1].split("auth_seq_num"))

    totaling_combinations = list()
    for table_name_prefix_suffix in aut_seq_all_splitted:
        combinations = list()
        for key in mmcif_dict_keys:
            if table_name_prefix_suffix[0] == key.split(".")[0]:
                # res_num auth_seq_id or auth_seq_num
                if table_name_prefix_suffix[1] in key and table_name_prefix_suffix[2] in key \
                        and "auth_seq_id" in key or "auth_seq_num" in key:
                    combinations.append(key)
                # chain auth_asym_id or strand_id
                if "assembly" in mmcif_dict["data_"]:
                    if table_name_prefix_suffix[1] in key and table_name_prefix_suffix[2] in key \
                            and "orig_auth_asym_id" in key:
                        combinations.append(key)
                else:
                    if table_name_prefix_suffix[1] in key and table_name_prefix_suffix[2] in key \
                            and "auth_asym_id" in key or "strand_id" in key:
                        combinations.append(key)
                # ins_code
                if table_name_prefix_suffix[1] in key and table_name_prefix_suffix[2] in key \
                        and "ins_code" in key:
                    combinations.append(key)
                # monomer_type or auth_comp_id or auth_mon_id or mon_id for _struct_ref_seq_dif
                if table_name_prefix_suffix[1] in key and table_name_prefix_suffix[2] in key \
                        and "auth_comp_id" in key or "auth_mon_id" in key:
                    combinations.append(key)
                elif table_name_prefix_suffix[0] == "_struct_ref_seq_dif" \
                        and "mon_id" in key and "db_mon_id" not in key:
                    combinations.append(key)

        # work assuming all the elements in right order
        # and they are not crossing each other
        if len(combinations) > 4:
            combinations = combinations[:4]

        ordered_combination = list()
        for name in combinations:
            if "auth_seq" in name:
                ordered_combination.insert(0, name)
        for name in combinations:
            if "auth_asym_id" in name or "strand_id" in name:
                ordered_combination.insert(1, name)
        for name in combinations:
            if "ins_code" in name:
                ordered_combination.insert(2, name)
        for name in combinations:
            if "auth_comp_id" in name or "mon_id" in name:
                ordered_combination.insert(3, name)

        # exceptions
        if (  # "pdbx_unobs_or_zero_occ_residues" not in ordered_combination[0]
                "nonpoly_scheme" not in ordered_combination[0]
                and "poly_seq_scheme" not in ordered_combination[0]
                and "ndb_struct_na_base" not in ordered_combination[0]):
            totaling_combinations.append(ordered_combination)

    return totaling_combinations
    
def renumber_tables(formed_columns, mmcif_dict, poly_nonpoly_atom_site, chains_to_change, default_num):
    dot_or_question_tuple = (".", "?")
    for n in formed_columns:
        auth_comp_id = 0
        auth_seq_id = n[0]
        auth_asym_id = n[1]
        try:
            PDB_ins_code = n[2]
            if "ins_code" not in PDB_ins_code:
                auth_comp_id = PDB_ins_code
                PDB_ins_code = 0
        except IndexError:
            PDB_ins_code = 0
        try:
            if auth_comp_id == 0:
                auth_comp_id = n[3]
        except IndexError:
            auth_comp_id = 0

        if "_pdbx_branch_scheme" in auth_seq_id:
            auth_seq_id = "_pdbx_branch_scheme.pdb_seq_num"
            auth_asym_id = "_pdbx_branch_scheme.pdb_asym_id"

        PDB_ins_code_list = list()
        # auth_comp_id_list = mmcif_dict[auth_comp_id] #for debug only
        auth_seq_id_list = mmcif_dict[auth_seq_id]
        auth_asym_id_list = mmcif_dict[auth_asym_id]

        if PDB_ins_code == 0:
            for _ in range(len(auth_seq_id_list)):
                PDB_ins_code_list.append("?")
        else:
            PDB_ins_code_list = mmcif_dict[PDB_ins_code]

        if type(auth_asym_id_list) == str:
            # auth_comp_id_list = [auth_comp_id_list] for debug only
            auth_seq_id_list = [auth_seq_id_list]
            auth_asym_id_list = [auth_asym_id_list]

            if PDB_ins_code == 0:
                PDB_ins_code_list = ["?"]
            else:
                PDB_ins_code_list = [PDB_ins_code]

        if PDB_ins_code != 0:
            dot_to_question = list()
            for ins_code in mmcif_dict[PDB_ins_code]:
                if ins_code == ".":
                    dot_to_question.append("?")
                else:
                    dot_to_question.append(ins_code)
            PDB_ins_code_list = dot_to_question

        auth_seq_id_list_zip = list(zip(auth_seq_id_list, auth_asym_id_list))
        df_mmCIF_auth_seq_id_list_zip = pd.DataFrame(zip(auth_seq_id_list_zip, PDB_ins_code_list))
        df_mmCIF_auth_seq_id_list_zip = df_mmCIF_auth_seq_id_list_zip.rename(columns={0: "auth_seq_id_list_zip", 1: "ins_code"})

        df_mmCIF_auth_seq_id_list_zip["PDB_with_ins_code"] = np.where(df_mmCIF_auth_seq_id_list_zip['ins_code'] != "?",
                                                                      (df_mmCIF_auth_seq_id_list_zip['auth_seq_id_list_zip'].apply(lambda x: x[0])
                                                                       + df_mmCIF_auth_seq_id_list_zip['ins_code'].apply(lambda y: y[0]) + ","
                                                                       + df_mmCIF_auth_seq_id_list_zip['auth_seq_id_list_zip'].apply(lambda x: x[1])),
                                                                      df_mmCIF_auth_seq_id_list_zip['ins_code'])

        df_mmCIF_auth_seq_id_list_zip["PDB_with_ins_code_cor"] = np.where(df_mmCIF_auth_seq_id_list_zip['PDB_with_ins_code'] != "?",
                                                                          df_mmCIF_auth_seq_id_list_zip["PDB_with_ins_code"].apply(
                                                                              lambda x: tuple(x.split(","))),
                                                                          df_mmCIF_auth_seq_id_list_zip["auth_seq_id_list_zip"])

        df_mmCIF_auth_seq_id_list_zip["auth_seq_id_list_zip"] = df_mmCIF_auth_seq_id_list_zip["PDB_with_ins_code_cor"]
        df_mmCIF_auth_seq_id_list_zip = df_mmCIF_auth_seq_id_list_zip.drop(columns=["PDB_with_ins_code_cor", "ins_code", "PDB_with_ins_code"])

        df_auth_seq_id_list_zip_final = df_mmCIF_auth_seq_id_list_zip.merge(poly_nonpoly_atom_site, left_on="auth_seq_id_list_zip",
                                                                            right_on="PDB_num_and_chain", how='left')

        df_auth_seq_id_list_zip_final["question_mark"] = np.where(
            df_auth_seq_id_list_zip_final["auth_seq_id_list_zip"].apply(lambda x: x[0] in dot_or_question_tuple),
            df_auth_seq_id_list_zip_final["auth_seq_id_list_zip"].apply(lambda x: x[0]),
            df_auth_seq_id_list_zip_final["Uni_or_50k"].apply(lambda x: x))
        try:
            df_auth_seq_id_list_zip_final["final"] = np.where(df_auth_seq_id_list_zip_final["question_mark"].apply(lambda x: type(x) == float),
                                                              df_auth_seq_id_list_zip_final["auth_seq_id_list_zip"].apply(
                                                                  lambda x: "." if x[0] == "." else
                                                                  "?" if x[0] == "?" else str(
                                                                      int(''.join(filter(str.isdigit, str(x[0])))) + default_num)
                                                                  if x[1] in chains_to_change else str(int(''.join(filter(str.isdigit, str(x[0])))))),
                                                              df_auth_seq_id_list_zip_final["question_mark"].apply(lambda x: x))
        except ValueError:
            # print("ValueError in table " + auth_seq_id + " has non-numeric value point in file " + mmcif_dict["data_"])
            return print("ValueError in table " + auth_seq_id + " has non-numeric value point in file " + mmcif_dict["data_"])

        df_auth_seq_id_list_zip_final["ins_code"] = df_auth_seq_id_list_zip_final["final"].apply(lambda x: "?"
        if re.sub('[0-9]+', '', x).strip("-").strip(".").strip('?') == ""
        else re.sub('[0-9]+', '', x).strip("-").strip(".").strip('?'))
        df_auth_seq_id_list_zip_final["final"] = df_auth_seq_id_list_zip_final["final"].apply(lambda x: x.strip(re.sub('[0-9\-\?\.]+', '', x)))

        for num in df_auth_seq_id_list_zip_final["final"]:
            if num == "":
                print("Empty str")
            if type(num) == float:
                print("Float or npNAN")

        # actual replacing auth_num with UniProt_num and of ins_code with '?'

        PDB_ins_code_list = list()
        if PDB_ins_code != 0:
            if "." in mmcif_dict[PDB_ins_code]:
                for ins in df_auth_seq_id_list_zip_final["ins_code"].values:
                    if "?" == ins:
                        PDB_ins_code_list.append(".")
                    else:
                        PDB_ins_code_list.append(ins)
                mmcif_dict[PDB_ins_code] = PDB_ins_code_list
            else:
                mmcif_dict[PDB_ins_code] = list(df_auth_seq_id_list_zip_final["ins_code"].values)

        if "_pdbx_branch_scheme" in auth_seq_id:
            mmcif_dict["_pdbx_branch_scheme.auth_seq_num"] = list(df_auth_seq_id_list_zip_final["final"].values)
        else:
            mmcif_dict[auth_seq_id] = list(df_auth_seq_id_list_zip_final["final"].values)

    return mmcif_dict

