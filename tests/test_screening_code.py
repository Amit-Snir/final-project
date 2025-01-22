import sys
import os

#adding path for the source
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import pandas as pd
import numpy as np
#import the functions
from screening_code import validate_and_format_columns, check_for_non_numeric_data, filter_missing_data, fill_missing_data, save_updated_file

class TestScreeningCode(unittest.TestCase):
    
    def setUp(self):
        #making dataset for cheking sake
        data = {
            'timestamps': [1, 2, 3, 4, 5],
            'TP9': [0.5, 0.6, 0.7, 0.8, 0.9],
            'AF7': [1.1, 1.2, 1.3, 1.4, 1.5],
            'AF8': [2.1, 2.2, 2.3, 2.4, 2.5],
            'TP10': [3.1, 3.2, 3.3, 3.4, 3.5]
        }
        self.df = pd.DataFrame(data)

        #making a temporary csv file for checking it
        self.test_file_path = 'test_subject.csv'
        self.df.to_csv(self.test_file_path, index=False)
    
    def tearDown(self):
        # cleaning the csv after each test
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_validate_and_format_columns(self):
        df_valid = validate_and_format_columns(self.df.copy())
        self.assertIsNotNone(df_valid)
        
        #adding wrong column name
        df_invalid = self.df.copy()
        df_invalid.rename(columns={'TP9': 'InvalidColumn'}, inplace=True)
        df_invalid_result = validate_and_format_columns(df_invalid)
        self.assertIsNone(df_invalid_result)
    
    def test_check_for_non_numeric_data(self):
        df_valid = check_for_non_numeric_data(self.df.copy())
        self.assertIsNotNone(df_valid)
        
        #adding a none numeric value
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'TP9'] = 'NaN'
        df_invalid_result = check_for_non_numeric_data(df_invalid)
        self.assertIsNone(df_invalid_result)
    
    def test_filter_missing_data(self):
        df_invalid = pd.DataFrame({
            'timestamps': [1, 2, 3, 4, 5],
            'TP9': [1.0, 2.0, 3.0, 4.0, 5.0],
            'AF7': [1.0, 2.0, 3.0, 4.0, 5.0],
            'AF8': [1.0, 2.0, 3.0, 4.0, 5.0],
            'TP10': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        df_valid = filter_missing_data(df_invalid)
        self.assertIsNone(df_valid)

        df_valid_large = pd.DataFrame({
            'timestamps': np.arange(1, 11001),
            'TP9': np.random.rand(11000),
            'AF7': np.random.rand(11000),
            'AF8': np.random.rand(11000),
            'TP10': np.random.rand(11000)
        })

        df_valid = filter_missing_data(df_valid_large)
        self.assertIsNotNone(df_valid)

    def test_fill_missing_data(self):
        df_valid = self.df.copy()
        df_valid.loc[0, 'TP9'] = np.nan  #adding missing value
        df_filled = fill_missing_data(df_valid)
        self.assertIsNotNone(df_filled)
        self.assertEqual(df_filled['TP9'].isna().sum(), 0)  #making sure no missing values remains
    
    def test_save_updated_file(self):
        file_path = 'test_output.csv'
        save_updated_file(self.df, file_path)
        self.assertTrue(os.path.exists(file_path))  #making sure the file is being saved
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
