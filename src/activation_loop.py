import re

def parse_activation_loops(fasta_file):
    """
    Parses a FASTA file containing activation loop ranges and returns a dictionary
    with gene groups as keys and activation loop ranges as values.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        dict: A dictionary with gene groups as keys and activation loop ranges as values.
    """
    # Initialize an empty dictionary
    activation_loops = {}

    # Open the FASTA file
    with open(fasta_file, "r") as file:
        for line in file:
            # Check if the line is a header (starts with '>')
            if line.startswith(">"):
                # Use a regular expression to find the activation loop range (e.g., 292-319)
                match_range = re.search(r"\s(\d+-\d+)\s", line)
                # Use a regular expression to find the gene group (e.g., AGC_AKT1)
                match_group = re.search(r"^>(\w+(-\d+)?)", line)

                # If matches are found, add the gene group and activation loop range to the dictionary
                if match_range and match_group:
                    gene_group = match_group.group(1)
                    activation_loop_range = match_range.group(1)
                    activation_loops[gene_group] = activation_loop_range

    return activation_loops

if __name__ == "__main__":
    # Usage example:
    fasta_file = "actloop.fasta"
    activation_loops = parse_activation_loops(fasta_file)
    print(activation_loops)

