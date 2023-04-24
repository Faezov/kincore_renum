import os

def clean_up_files(structure2_name):
    for file in os.listdir():
        if structure2_name in file:
            os.remove(file)

def remove_files_not_ending_with(directory, suffix):
    for file in os.listdir(directory):
        if not file.endswith(suffix):
            os.remove(os.path.join(directory, file))
