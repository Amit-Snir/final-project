import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
import sys
import os
from unittest.mock import MagicMock
import matplotlib.pyplot as plt

#path for source code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from new_predicted_state import (
    load_training_data, load_participant_data, transform_participant_data, predict_emotional_state
)

class TestEEGPrediction(unittest.TestCase):

    def setUp(self):
        # Create mock training and participant datasets
        self.mock_training_data = pd.DataFrame({
            'TP9': [10, 20, 30],
            'TP10': [15, 25, 35],
            'AF8': [5, 15, 25],
            'AF7': [10, 20, 30],
            'wave': ['ALPHA', 'BETA', 'DELTA'],
            'Emotional_State': ['C', 'N', 'R']
        })

        self.mock_participant_data = pd.DataFrame({
            'Electrode': ['TP9', 'TP10', 'AF8', 'AF7'],
            'Wave Type': ['ALPHA', 'BETA', 'DELTA', 'GAMMA'],
            'Outliers Count': [10, 20, 30, 40]
        })

    def test_load_training_data(self):
        # Test valid training data loading
        with patch('pandas.read_excel', return_value=self.mock_training_data):
            result = load_training_data("mock_path.xlsx")
            self.assertIsNotNone(result, "Should load valid training data correctly")
            self.assertTrue('Emotional_State' in result.columns, "Emotional_State column should exist")

    def test_load_participant_data(self):
        # Test valid participant data loading
        with patch('pandas.read_csv', return_value=self.mock_participant_data):
            result = load_participant_data("mock_path.csv")
            self.assertIsNotNone(result, "Should load valid participant data correctly")
            self.assertTrue('Electrode' in result.index.names or 'Electrode' in result.columns,
                            "Electrode column should exist after transformation")

    def test_predict_emotional_state(self):
        # Test prediction process with valid data
        from sklearn.ensemble import RandomForestClassifier

        # Mock the trained model
        model = RandomForestClassifier()
        model.fit(self.mock_training_data[['TP9', 'TP10', 'AF8', 'AF7']],
                  self.mock_training_data['Emotional_State'])

        participant_data_aligned = pd.DataFrame({
            'TP9': [10], 'TP10': [20], 'AF8': [30], 'AF7': [40]
        })

        with patch('pandas.read_csv', return_value=participant_data_aligned):
            # Ensure no exception occurs
            try:
                predict_emotional_state(model, "mock_path.csv")
                # Check if a figure is created
                fig = plt.gcf()  # Get current figure
                self.assertIsNotNone(fig, "A graph should be created.")
            except Exception as e:
                self.fail(f"predict_emotional_state raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
