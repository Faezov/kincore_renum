import os

def extract_path_components(fullpath):
    gg = os.path.basename(fullpath)
    tempconf, MSAlim, MSAsource = os.path.basename(os.path.dirname(fullpath)).split("_")
    return gg, tempconf, MSAlim, MSAsource
