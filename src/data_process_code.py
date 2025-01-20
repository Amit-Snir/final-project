import numpy as np
import pandas as pd
from scipy.signal import resample
import os
import matplotlib.pyplot as plt

# פונקציה לריסמפלינג של הנתונים (בין 150 ל-270Hz בצורה לא עקבית)
def resample_eeg_data(file_path, target_sampling_rate=200):
    try:
        df = pd.read_csv(file_path)
        
        # ודא שהקובץ מכיל את העמודות הנכונות
        columns = ['TP9', 'TP10', 'AF8', 'AF7']
        
        resampled_data = {}
        
        # ריסמפלינג לכל אחת מהאלקטרודות
        for column in columns:
            current_length = len(df[column])
            resampled_data[column] = resample(df[column], int(target_sampling_rate * (df['timestamps'].iloc[-1] - df['timestamps'].iloc[0])))

        # יצירת חותמת זמן חדשה מ-0 שניות
        new_timestamps = pd.Series(range(len(resampled_data['TP9']))) / target_sampling_rate
        resampled_df = pd.DataFrame(resampled_data)
        resampled_df.insert(0, 'timestamps', new_timestamps)

        # החזרת ה-DataFrame המעודכן
        return resampled_df
    
    except Exception as e:
        print(f"Error while resampling: {e}")
        return None

# פונקציה לחישוב EMA (ממוצע נע) להחלקה של הנתונים
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

# פונקציה להסרת אאוטליירס בעזרת Z-score (כאשר Z > 2)
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

# פונקציה למילוי ערכים חסרים
def fill_missing_values(df):
    try:
        cleaned_df = df.bfill(axis=0).ffill(axis=0)
        return cleaned_df
    except Exception as e:
        print(f"Error while filling missing value: {e}")
        return None

# פונקציה ליצירת היסטוגרמה עם האאוטליירס ושמירה של המידע לקובץ CSV
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

        # מילון לאחסון אאוטליירס לכל אלקטרודה
        outliers_per_wave = {col: [] for col in columns}
        file_name = os.path.basename(file_path)

        # יצירת גרף היסטוגרמה
        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.15

        # מילון לאחסון מספר האאוטליירס לכל אלקטרודה
        summary_data = {col: {wave: 0 for wave in wave_bands.keys()} for col in columns}

        # לולאה עבור כל אלקטרודה
        for i, column in enumerate(columns):
            signal = df[column].dropna()  # הסרת נתונים חסרים
            fs = 200  # תדר דגימה
            n = len(signal)
            freqs = np.fft.fftfreq(n, d=1/fs)  # חישוב התדרים
            fft_values = np.fft.fft(signal)
            magnitudes = np.abs(fft_values)  # חישוב העוצמה של כל תדר
            valid_freqs = (freqs >= 0.5) & (freqs <= 100)  # הגבלת טווח התדרים
            valid_magnitudes = magnitudes[valid_freqs]
            valid_freqs = freqs[valid_freqs]
            top_5_percent_idx = np.argsort(valid_magnitudes)[-int(0.05 * len(valid_magnitudes)):]  # 5% העליונים

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

        # יצירת DataFrame עם נתוני האאוטליירס
        outliers_summary = []
        for column in columns:
            for wave in wave_bands.keys():
                outliers_summary.append({'Electrode': column, 'Wave Type': wave, 'Outliers Count': summary_data[column][wave]})

        # יצירת DataFrame עם המידע
        summary_df = pd.DataFrame(outliers_summary)

        # שמירה לקובץ CSV
        summary_df.to_csv(r'C:\python advenced\final-project\data\3. passed_process_data\passed_process_data.csv', index=False)
        print(f"✅ data process done seccesfully")

    except Exception as e:
        print("Error while FFT and plotting top 5% magnitude outliers:", e)
