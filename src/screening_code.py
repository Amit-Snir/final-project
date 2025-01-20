import os
import pandas as pd
import numpy as np

file_path = r"C:\python advenced\my_final_reposatory\data\1. initial_subjects_data\subject_1.csv"

# file type screening
try:
    # checking wheather its a csv file or not
    if not file_path.endswith('.csv'):
        raise ValueError("The file is not in CSV format. The process will be terminated.")
    
    df = pd.read_csv(file_path)
    print(f"✅ File '{file_path}' has been successfully loaded.")
except ValueError as ve:
    print(f"❌ Error: {ve} - exiting code.")
    exit(1)  # not a csv, the file will fail the screening
except Exception as e:
    print(f"❌ Error while loading the file: {e} - exiting code.")
    exit(1)  # notice for a problem with loading the file

# the columns expected to be seen in the csv file
expected_columns = ['timestamps', 'TP9', 'AF7', 'AF8', 'TP10']

# function to validate the correct columns in the file
def validate_and_format_columns(df):
    try:
        # stripping spaces and upper all content of columns names
        df.columns = [col.strip().replace(' ', '').upper() for col in df.columns]

        # only "timestamps" is lower cased
        df.columns = [col.lower() if col == 'TIMESTAMPS' else col for col in df.columns]

        # comparing it to the expected coluns names
        if not all(col in df.columns for col in expected_columns):
            raise ValueError(f"The file has columns that do not match the expected columns.")
        
        print("✅ Column validation passed.")
        return df
    except ValueError as ve:
        print(f"❌ Error: {ve} - The file did not pass column validation. exiting code.")
        return None  # not all columns names correct - file failed screening
    except Exception as e:
        print(f"❌ Unexpected error: {e} - exiting code.")
        return None  # other error also will not pass screening

# checking all sampling data for non numeric data
def check_for_non_numeric_data(df):
    try:
        for row_idx, row in df.iterrows():
            for col_idx, value in row.items():
                # making all data numeric (float or negative)
                numeric_value = pd.to_numeric(value, errors='coerce')
                if pd.isna(numeric_value):  # if we get 'Nan' value after last stage
                    return None  

        print("✅ No non-numeric data found.")
        return df
    except Exception as e:
        print(f"❌ Unexpected error: {e} - exiting code.")
        return None  # non numeric - failed screening

# checking for missing data
def filter_missing_data(df):
    try:
        # counting number of rows in csv file
        num_rows = df.shape[0]
        print(f"🔍 Number of rows in the file: {num_rows}")
        # less than 11,000 rows it fails screening (aproximatly 45 seconds of data measuring)
        if num_rows < 11000:
            raise ValueError(f"❌ File has less than 11,000 rows ({num_rows} rows). This file is excluded.")
        # checking for missing values in timestamps column, if so - failed screening
        if df['timestamps'].isnull().sum() > 0:
            raise ValueError("❌ There are missing values in the 'timestamps' column. The file is excluded.")
        # checking for missing data - summing all no value cells. over 2,200 - failed screening (its more than 5% cells of 11,000 rows 4 electrodes)
        missing_samples = df.isnull().sum().sum()
        if missing_samples > 2200:
            raise ValueError(f"❌ More than 2200 missing samples ({missing_samples} samples). The file is excluded.")
        print(f"Total missing samples: {missing_samples}")
        print("✅ Missing data check passed.")
        return df
    except ValueError as ve:
        print(f"❌ Error: {ve} - The file did not pass missing data check. exiting code.")
        return None  # failed screening, exiting code
    except Exception as e:
        print(f"❌ Unexpected error: {e} - exiting code.")
        return None

# filling missing data - only for a file that passed the last stage
def fill_missing_data(df):
    try:
        missing_samples_before = df.isnull().sum().sum()
        if missing_samples_before > 0:  # checking for missing data
            print(f"🔄 Filling missing samples. Total missing samples before filling: {missing_samples_before}")
            df = df.fillna(method='ffill').fillna(method='bfill')
            missing_samples_after = df.isnull().sum().sum()
            print(f"Total missing samples after filling: {missing_samples_after}")
            # checking if there are still missing values after filling
            if missing_samples_after > 0:
                raise ValueError(f"❌ Not all missing values were filled. {missing_samples_after} samples are still missing.")
        else:
            print("✅ No missing data to fill.")  # reporting none data cells remain after filling 

        return df
    except ValueError as ve:
        print(f"{ve} - exiting code.")  # if a value error came up in the function's code, exiting it will happen
        return None 
    except Exception as e:
        print(f"❌ Unexpected error: {e} - exiting code.")  
        return None  

# saving updated file for sake of using it in the data_process stage (next code stage)
def save_updated_file(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        print(f"✅ File '{file_path}' passed screening succesfully.")
    except Exception as e:
        print(f"❌ Error while saving the file: {e} - exiting code.")

