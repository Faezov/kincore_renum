import os
import requests
import concurrent.futures
from tqdm import tqdm
from time import sleep
current_directory = os.getcwd()


def url_formation_for_pool(format_to_download="mmCIF", list_of_file_names=(),
                           default_input_path_to_mmCIF=current_directory + "/mmCIF",
                           default_input_path_to_PDB=current_directory + "/PDB",
                           default_input_path_to_SIFTS=current_directory + "/SIFTS",):
    """Forming URLs from PDB file names or PDB ids"""
    formats = {
        "mmCIF": {
            "input_path": default_input_path_to_mmCIF,
            "url_format": "https://files.rcsb.org/pub/pdb/data/structures/all/mmCIF/{target_name}.cif.gz"
        },
        "PDB": {
            "input_path": default_input_path_to_PDB,
            "url_format": "https://files.rcsb.org/pub/pdb/data/structures/all/pdb/pdb{target_name}.ent.gz",
        },
        "SIFTS": {
            "input_path": default_input_path_to_SIFTS,
            "url_format": "http://ftp.ebi.ac.uk/pub/databases/msd/sifts/xml/{target_name}.xml.gz",
        },
    }

    urls_to_target_files = [
        formats[format_to_download]["url_format"].format(
            target_name=file_name[3:7] if ".ent" in file_name and file_name.startswith("pdb") else file_name[0:4],
            file_name=file_name
        )
        for file_name in list_of_file_names
        if len(file_name) >= 4
    ]
    
    if format_to_download in formats:
        input_path = formats[format_to_download]["input_path"]
        if not os.path.exists(input_path):
            os.makedirs(input_path)

    return urls_to_target_files


def download_file(url, file_path, retries=2):
    """Downloading function with requests"""
    attempt = 0
    while attempt <= retries:
        try:
            r = requests.get(url, stream=True, timeout=10)
            if r.status_code == requests.codes.ok:
                with open(file_path, 'wb') as f:
                    for data in r.iter_content(chunk_size=1024):
                        f.write(data)
                break
        except Exception as e:
            print(f"Error downloading {url} on attempt {attempt + 1}: {str(e)}")
            if attempt < retries:
                sleep(5)  # Wait for 5 seconds before retrying
            attempt += 1

def download_with_pool(urls_to_target_files=(),
                       default_input_path_to_mmCIF=current_directory + "/mmCIF",
                       default_input_path_to_PDB=current_directory + "/PDB",
                       default_input_path_to_SIFTS=current_directory + "/SIFTS",):
    """Downloading in parallel with ThreadPoolExecutor"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for url in urls_to_target_files:
            file_name = url[url.rfind("/")+1:]
            format_of_db = url[url.rfind("/")-3:url.rfind("/")]

            if format_of_db == "CIF":
                file_path = os.path.join(default_input_path_to_mmCIF, file_name)
            elif format_of_db == "pdb":
                file_path = os.path.join(default_input_path_to_PDB, file_name)
            elif format_of_db == "xml":
                file_path = os.path.join(default_input_path_to_SIFTS, file_name)
            else:
                continue
                
            future = executor.submit(download_file, url, file_path)
            futures.append(future)
            
        # add progress bar
        for future in tqdm(concurrent.futures.as_completed(futures), 
                           total=len(futures), 
                           desc='Download '+format_of_db+" file"):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error downloading file: {str(e)}")
                continue
