import os
import concurrent.futures
from Bio.PDB import MMCIFParser, PDBIO, Select
from tqdm.notebook import tqdm

def convert_mmcif_to_pdb(input_directory, output_directory):
    input_directory_abspath = os.path.abspath(input_directory)
    output_directory_abspath = os.path.abspath(output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def _convert_mmcif_to_pdb(filename):
        mmcif_parser = MMCIFParser()
        pdb_io = PDBIO()

        if filename.endswith(".cif") or filename.endswith(".mmcif"):
            structure = mmcif_parser.get_structure("structure", os.path.join(input_directory_abspath, filename))
            clean_filename = filename.replace("_c100_", "_").replace("_plus_", "_").replace(".pdb", "")
            output_filename = os.path.splitext(clean_filename)[0] + ".pdb"
            pdb_io.set_structure(structure)
            pdb_io.save(os.path.join(output_directory_abspath, output_filename), Select())

    mmcif_files = [f for f in os.listdir(input_directory) if f.endswith(".cif") or f.endswith(".mmcif")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_convert_mmcif_to_pdb, filename) for filename in mmcif_files]

        try:
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(mmcif_files)):
                pass
        except KeyboardInterrupt:
            print("Terminating due to KeyboardInterrupt")
            for future in futures:
                future.cancel()
                
### Jupyter Notebook version                
# import os
# import concurrent.futures
# from Bio.PDB import MMCIFParser, PDBIO, Select
# from tqdm import tqdm

# # Set the input and output directories
# input_directory = "./structures/AF220_mmCIF_230317"
# output_directory = "./structures/AF220_PDB_230317"
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# def convert_mmcif_to_pdb(filename):
#     # Create a parser for mmCIF files
#     mmcif_parser = MMCIFParser()

#     # Create an object to write PDB files
#     pdb_io = PDBIO()

#     # Check if the file is an mmCIF file
#     if filename.endswith(".cif") or filename.endswith(".mmcif"):
#         # Parse the mmCIF file
#         structure = mmcif_parser.get_structure("structure", os.path.join(input_directory, filename))

#         # Remove redundant information from the filename
#         clean_filename = filename.replace("_c100_", "_").replace("_plus_", "_").replace(".pdb", "")

#         # Set the output filename
#         output_filename = os.path.splitext(clean_filename)[0] + ".pdb"

#         # Write the structure to a PDB file
#         pdb_io.set_structure(structure)
#         pdb_io.save(os.path.join(output_directory, output_filename), Select())

# # Create a list of all mmCIF files in the input directory
# mmcif_files = [f for f in os.listdir(input_directory) if f.endswith(".cif") or f.endswith(".mmcif")]


# # Use a thread pool to process the files concurrently
# with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
#     # Submit the tasks to the executor and store the Future objects in a list
#     futures = [executor.submit(convert_mmcif_to_pdb, filename) for filename in mmcif_files]

#     try:
#         # Iterate through the completed tasks with a tqdm progress bar
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(mmcif_files)):
#             # Do something with the result, if needed (in this case, we don't need the result)
#             pass

#     except KeyboardInterrupt:
#         print("Terminating due to KeyboardInterrupt")
#         # Cancel all the pending tasks
#         for future in futures:
#             future.cancel()
