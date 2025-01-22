import sys
import os

#path for source code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import pandas as pd
import numpy as np
from data_process_code import resample_eeg_data, calculate_ema, remove_outliers_z, fill_missing_values, plot_histogram_with_outliers_and_save

class TestDataProcessCode(unittest.TestCase):
    
    def setUp(self):
        #making data for checking sake
        data = {
            'timestamps': [1, 2, 3, 4, 5],
            'TP9': [0.5, 0.6, 0.7, 0.8, 0.9],
            'AF7': [1.1, 1.2, 1.3, 1.4, 1.5],
            'AF8': [2.1, 2.2, 2.3, 2.4, 2.5],
            'TP10': [3.1, 3.2, 3.3, 3.4, 3.5]
        }
        self.df = pd.DataFrame(data)

        #making a temporary csv file
        self.test_file_path = 'test_subject.csv'
        self.df.to_csv(self.test_file_path, index=False)
    
    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_resample_eeg_data(self):
        df_resampled = resample_eeg_data(self.test_file_path)
        self.assertIsNotNone(df_resampled)
        self.assertEqual(df_resampled.shape[0], len(df_resampled['TP9']))  #making sure the data size matches

    def test_calculate_ema(self):
        df_ema = calculate_ema(self.df.copy())
        self.assertIsNotNone(df_ema)
        self.assertEqual(df_ema.shape, self.df.shape)  #making sure the data size matches

    def test_remove_outliers_z(self):
        df_no_outliers, outliers = remove_outliers_z(self.df.copy())
        self.assertIsNotNone(df_no_outliers)
        self.assertIsInstance(outliers, dict)
        #makign sure each column is a list even if empty
        for col, values in outliers.items():
            self.assertIsInstance(values, list)
            if len(values) > 0:
                print(f"Outliers found for {col}: {values}")

    def test_fill_missing_values(self):
        df_with_missing = self.df.copy()
        df_with_missing.loc[0, 'TP9'] = np.nan  #adding missing value
        df_filled = fill_missing_values(df_with_missing)
        self.assertIsNotNone(df_filled)
        self.assertEqual(df_filled['TP9'].isna().sum(), 0)  #making sure no missing values exist

    def test_plot_histogram_with_outliers_and_save(self):
        file_path = 'test_subject.csv'
        plot_histogram_with_outliers_and_save(self.df.copy(), file_path)
        self.assertTrue(os.path.exists('C:\\python advenced\\final-project\\data\\3. passed_process_data\\passed_process_data.csv'))
    
if __name__ == '__main__':
    unittest.main()
