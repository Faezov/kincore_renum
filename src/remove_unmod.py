import os

def remove_unmodified_files(pdb_folder2):    
    for file in os.listdir(pdb_folder2):
        if "_unmodified" in file:
            os.remove(os.path.join(pdb_folder2, file))
