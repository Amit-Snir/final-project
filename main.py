import sys
import pandas as pd

sys.path.append(r"C:\python advenced\final-project\src")

from screening_code import (
    validate_and_format_columns,
    check_for_non_numeric_data,
    filter_missing_data,
    fill_missing_data,
    save_updated_file
)

file_path = r"C:\python advenced\final-project\data\1. initial_subjects_data\subject_3.csv"
if not file_path.endswith('.csv'): exit(1)

df = pd.read_csv(file_path)
df = validate_and_format_columns(df)
df = filter_missing_data(df)
df = check_for_non_numeric_data(df)
df = fill_missing_data(df)

save_updated_file(df, r"C:\python advenced\final-project\data\2. passed_screening_data\subject_3.csv")

#____________________________________________________________________________________________________________________________________#

sys.path.append(r"C:\python advenced\my_final_reposatory\src")

from data_process_code import (
    resample_eeg_data,
    calculate_ema,
    remove_outliers_z,
    fill_missing_values,
    plot_histogram_with_outliers_and_save
)

#what file to use - file of source
file_path = r"C:\python advenced\final-project\data\2. passed_screening_data\subject_3.csv"

df_resampled = resample_eeg_data(file_path)
df_ema = calculate_ema(df_resampled)
df_no_outliers, outliers_info = remove_outliers_z(df_ema)
df_filled = fill_missing_values(df_no_outliers)

plot_histogram_with_outliers_and_save(df_filled, file_path)

#____________________________________________________________________________________________________________________________________#

#import code
sys.path.append(r"C:\python advenced\final-project\src")

#import dunction from code
from new_predicted_state import main

#running "main" function
if __name__ == "__main__":
    main()

