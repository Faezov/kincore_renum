import os
import re
import datetime
import pandas as pd

def get_latest_result_df():
    # List all the files in the current working directory
    files = os.listdir()

    # Filter files with the correct pattern using a regex
    pattern = re.compile(r"results_df_(\d{6})\.txt")
    filtered_files = [f for f in files if pattern.match(f)]

    # Find the file with the latest date
    latest_file = None
    latest_date = None

    for file_name in filtered_files:
        date_str = pattern.match(file_name).group(1)
        file_date = datetime.datetime.strptime(date_str, "%y%m%d")

        if latest_date is None or file_date > latest_date:
            latest_date = file_date
            latest_file = file_name

    with open(latest_file, 'r') as file:
        results_df_loaded = pd.read_csv(file, sep='\s+')
        
    return results_df_loaded
