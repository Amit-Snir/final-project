import numpy as np
import pandas as pd
from scipy.signal import resample
import os
import matplotlib.pyplot as plt

#resample the data that was sampled inconsistently between 150 - 270 Hz into 200 Hz consistent
def resample_eeg_data(file_path, target_sampling_rate=200):
    try:
        df = pd.read_csv(file_path)
        columns = ['TP9', 'TP10', 'AF8', 'AF7']
        resampled_data = {}
        #resample each of the electrodes
        for column in columns:
            current_length = len(df[column])
            resampled_data[column] = resample(df[column], int(target_sampling_rate * (df['timestamps'].iloc[-1] - df['timestamps'].iloc[0])))
        #creating new timestamps starting from 0 seconds with 1/200 sec defference between each sample
        new_timestamps = pd.Series(range(len(resampled_data['TP9']))) / target_sampling_rate
        resampled_df = pd.DataFrame(resampled_data)
        resampled_df.insert(0, 'timestamps', new_timestamps)
        return resampled_df
    
    except Exception as e:
        print(f"Error while resampling: {e}")
        return None

#smoothing the data with EMA with Alpha of 0.1 value (between 0.1-0.3 is acceptable)
def calculate_ema(df, alpha=0.1):
    try:
        columns = ['TP9', 'TP10', 'AF8', 'AF7']
        ema_data = {}
        for column in columns:
            ema_data[column] = df[column].ewm(alpha=alpha).mean()
        ema_df = pd.DataFrame(ema_data)
        ema_df.insert(0, 'timestamps', df['timestamps'])
        return ema_df
    except Exception as e:
        print(f"Error while calculating EMA: {e}")
        return None

#removing outliers samples above 2 SD for it being probably noise
def remove_outliers_z(df, threshold=2):
    try:
        columns = ['TP9', 'TP10', 'AF8', 'AF7']
        outliers_per_wave = {col: [] for col in columns}
        for column in columns:
            ema = df[column].ewm(alpha=0.1).mean()
            mean = np.mean(ema)
            std_dev = np.std(ema)
            z_scores = (ema - mean) / std_dev
            outliers = np.abs(z_scores) > threshold
            df.loc[outliers, column] = np.nan  # שימוש ב.loc כדי למנוע שגיאת ChainedAssignment
            outliers_per_wave[column] = df[column][outliers].dropna().tolist()
        return df, outliers_per_wave
    except Exception as e:
        print(f"Error while removing outliers: {e}")
        return None, None

#filling missing values from last stage with Bfil and Ffil - just for it to be consistent
def fill_missing_values(df):
    try:
        cleaned_df = df.bfill(axis=0).ffill(axis=0)
        return cleaned_df
    except Exception as e:
        print(f"Error while filling missing value: {e}")
        return None

# plotting final outliers from FFT for histogram plotting
def plot_histogram_with_outliers_and_save(df, file_path):
    try:
        columns = ['TP9', 'TP10', 'AF8', 'AF7']
        colors = ['b', 'g', 'r', 'c']
        wave_bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (13, 30),
            'Gamma': (30, 100)
        }
        outliers_per_wave = {col: [] for col in columns}
        file_name = os.path.basename(file_path)
        #subplotting all 4 electrodes together and for each all of the wave lengths ranges
        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.15
        #collecting outliers from FFT distribution
        summary_data = {col: {wave: 0 for wave in wave_bands.keys()} for col in columns}
        #collecting from each electrode
        for i, column in enumerate(columns):
            signal = df[column].dropna()  #dropping empty values
            fs = 200  #sampling freq
            n = len(signal)
            freqs = np.fft.fftfreq(n, d=1/fs)  #calculating freq
            fft_values = np.fft.fft(signal)
            magnitudes = np.abs(fft_values)  #calculating magnitude for each freq sample
            valid_freqs = (freqs >= 0.5) & (freqs <= 100)  #limiting range of freqs to logicas brain pruduced freq (0/5-100)
            valid_magnitudes = magnitudes[valid_freqs]
            valid_freqs = freqs[valid_freqs]
            top_5_percent_idx = np.argsort(valid_magnitudes)[-int(0.05 * len(valid_magnitudes)):]  #using only significant top 5% magnitude outliers

            top_frequencies = valid_freqs[top_5_percent_idx]
            top_magnitudes = valid_magnitudes[top_5_percent_idx]

            outliers_per_wave[column] = top_frequencies.tolist()

            counts = {wave: 0 for wave in wave_bands.keys()}
            for wave, (f_min_band, f_max_band) in wave_bands.items():
                outliers_in_band = [f for f in outliers_per_wave[column] if f_min_band <= f <= f_max_band]
                counts[wave] = len(outliers_in_band)

                summary_data[column][wave] = counts[wave]

            x_positions = np.arange(len(wave_bands)) + i * width
            sorted_counts = [counts[wave] for wave in wave_bands.keys()]
            ax.bar(x_positions, sorted_counts, width=width, color=colors[i], alpha=0.6, label=f'{column} Outliers')

        ax.set_title(f'No of top 5% magnitude Outliers by Wave Type for Each Electrode - {file_name}')
        ax.set_xlabel('Wave Type')
        ax.set_ylabel('Number of Outliers')
        ax.set_xticks(np.arange(len(wave_bands)) + width * 2)
        ax.set_xticklabels(wave_bands.keys())
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        #df from last calculation
        outliers_summary = []
        for column in columns:
            for wave in wave_bands.keys():
                outliers_summary.append({'Electrode': column, 'Wave Type': wave, 'Outliers Count': summary_data[column][wave]})
        summary_df = pd.DataFrame(outliers_summary)

        #saving the df into csv file for next stage sake and printing succesfull finish confirmation
        summary_df.to_csv(r'C:\python advenced\final-project\data\3. passed_process_data\passed_process_data.csv', index=False)
        print(f"✅ data process done seccesfully")

    except Exception as e:
        print("Error while FFT and plotting top 5% magnitude outliers:", e)
