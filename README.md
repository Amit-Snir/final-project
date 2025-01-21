# final-project

This Python code is designed to process EEG data recorded using the MUSE device and predict the mental state (e.g., concentrating, relaxing) of the subject based on the recorded signals. The code performs several steps including data preprocessing, feature extraction, and training a machine learning model to predict the emotional state. the process follows the below steps:

1. Initial Data Requirements:
Input Data:

The input data should be EEG recordings from the MUSE device. These files should be in CSV format and contain the following columns:
timestamps: The time points for each EEG sample.
TP9, TP10, AF7, AF8: The EEG readings from the MUSE device's 4 electrodes (TP9, TP10, AF7, AF8).
Example File Names:

subjecta-concentrating-1.csv
subjectb-relaxed-1.csv
Each file should include the EEG measurements at varying sample rates, typically between 170 and 250 samples per second.

2. Step-by-Step Code Execution:
Step 1: Preprocessing the Raw EEG Data
Step 2: Resampling the Data to 200 Hz
Step 3: Smoothing the Data with Exponentially Moving Average (EMA)
Step 4: Feature Extraction
Step 5: Training the Random Forest Model
Step 6: Evaluating the Model
Step 7: Prediction

3. Final Output:
The final output will be a predicted mental state (e.g., "concentrating" or "relaxed") based on the EEG signals. This is achieved after the data has been preprocessed, features have been extracted, and a machine learning model has been trained to classify the emotional state.



