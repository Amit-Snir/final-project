import pandas as pd
from scipy.stats import f_oneway, pearsonr, spearmanr, ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    #Load the cleaned dataset.
    return pd.read_excel(file_path, engine="openpyxl")

def cohens_d(group1, group2):
    #Calculate Cohen's d effect size.
    try:
        if len(group1) < 2 or len(group2) < 2:
            return np.nan
        mean1, mean2 = np.mean(group1), np.mean(group2)
        pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
        if pooled_std == 0:
            return np.nan
        return (mean1 - mean2) / pooled_std
    except Exception as e:
        return np.nan

def perform_anova_ttest(data, wave_types, electrodes):
    #Perform ANOVA and T-Test for each wave type and electrode.
    anova_results = []
    ttest_results = []

    for wave in wave_types:
        for electrode in electrodes:
            try:
                wave_c = data[(data['State'] == 'C') & (data['wave'] == wave)][electrode].dropna()
                wave_r = data[(data['State'] == 'R') & (data['wave'] == wave)][electrode].dropna()
                wave_n = data[(data['State'] == 'N') & (data['wave'] == wave)][electrode].dropna()

                if len(set(wave_c)) > 1 and len(set(wave_r)) > 1 and len(set(wave_n)) > 1:
                    f_stat, p_value = f_oneway(wave_c, wave_r, wave_n)
                    anova_results.append({
                        'Wave': wave,
                        'Electrode': electrode,
                        'F-Statistic': f_stat,
                        'p-Value': p_value,
                        "Cohen's d (C vs R)": cohens_d(wave_c, wave_r),
                        "Cohen's d (C vs N)": cohens_d(wave_c, wave_n),
                        "Cohen's d (R vs N)": cohens_d(wave_r, wave_n)
                    })

                if len(wave_c) > 1 and len(wave_r) > 1:
                    t_stat, p_value = ttest_ind(wave_c, wave_r)
                    ttest_results.append({'Wave': wave, 'Electrode': electrode, 'Comparison': 'C vs R', 'T-Statistic': t_stat, 'p-Value': p_value})

                if len(wave_c) > 1 and len(wave_n) > 1:
                    t_stat, p_value = ttest_ind(wave_c, wave_n)
                    ttest_results.append({'Wave': wave, 'Electrode': electrode, 'Comparison': 'C vs N', 'T-Statistic': t_stat, 'p-Value': p_value})

                if len(wave_r) > 1 and len(wave_n) > 1:
                    t_stat, p_value = ttest_ind(wave_r, wave_n)
                    ttest_results.append({'Wave': wave, 'Electrode': electrode, 'Comparison': 'R vs N', 'T-Statistic': t_stat, 'p-Value': p_value})
            except Exception as e:
                print(f"Error processing {wave} - {electrode}: {e}")
    
    return pd.DataFrame(anova_results), pd.DataFrame(ttest_results)

def analyze_correlations(data, electrodes):
    #Analyze correlations between electrodes for each state.
    correlation_results = []

    for state in ['C', 'R', 'N']:
        state_data = data[data['State'] == state][electrodes]
        try:
            for col1 in electrodes:
                for col2 in electrodes:
                    if col1 != col2:
                        try:
                            pearson_corr, pearson_p_value = pearsonr(state_data[col1].dropna(), state_data[col2].dropna())
                            spearman_corr, spearman_p_value = spearmanr(state_data[col1].dropna(), state_data[col2].dropna())
                            correlation_results.append({
                                'State': state,
                                'Electrode 1': col1,
                                'Electrode 2': col2,
                                'Pearson Correlation': pearson_corr,
                                'Pearson P-Value': pearson_p_value,
                                'Spearman Correlation': spearman_corr,
                                'Spearman P-Value': spearman_p_value
                            })
                        except Exception as e:
                            print(f"Error calculating correlation for {state}, {col1}, {col2}: {e}")
        except Exception as e:
            print(f"Error calculating correlations for {state}: {e}")

    return pd.DataFrame(correlation_results)

def plot_outliers(data, wave_types, electrodes):
    #Plot outlier counts for each wave and electrode.
    for wave in wave_types:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{wave} Waves - Outliers by State and Electrode', fontsize=16)

        for idx, electrode in enumerate(electrodes):
            ax = axs[idx // 2, idx % 2]
            wave_means = data[data['wave'] == wave].groupby('State')[electrode].mean()
            if not wave_means.empty:
                wave_means.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'{electrode}', fontsize=14)
                ax.set_ylabel('Number of Outliers')
                ax.set_xlabel('Emotional State')
                ax.grid(axis='y')
            else:
                ax.set_title(f'{electrode} (No Data)', fontsize=14)
                ax.set_axis_off()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def plot_correlation_heatmaps_from_results(correlation_df, electrodes):
    #Plot correlation heatmaps using precomputed correlation results.
    for state in ['C', 'R', 'N']:
        plt.figure(figsize=(10, 8))
        plt.title(f"Correlation Heatmap for State: {state}")

        # Filter the correlations for the current state
        state_correlations = correlation_df[correlation_df['State'] == state]

        # Create a correlation matrix for the heatmap
        correlation_matrix = pd.DataFrame(index=electrodes, columns=electrodes, dtype=float)
        for _, row in state_correlations.iterrows():
            correlation_matrix.loc[row['Electrode 1'], row['Electrode 2']] = row['Pearson Correlation']
            correlation_matrix.loc[row['Electrode 2'], row['Electrode 1']] = row['Pearson Correlation']

        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)
        plt.show()

def main():
    file_path = r"C:\python advenced\final-project\data\cleaned_after_processing.xlsx"
    wave_types = ['DELTA', 'THETA', 'ALPHA', 'BETA', 'GAMMA']
    electrodes = ['TP9', 'TP10', 'AF8', 'AF7']

    data = load_data(file_path)

    anova_df, ttest_df = perform_anova_ttest(data, wave_types, electrodes)
    print("\nFinal Summary of ANOVA Results with Cohen's d:")
    print(anova_df)

    print("\nFinal Summary of T-Test Results:")
    print(ttest_df)

    correlation_df = analyze_correlations(data, electrodes)
    print("\nFinal Summary of Correlation Results:")
    print(correlation_df)

    plot_outliers(data, wave_types, electrodes)
    plot_correlation_heatmaps_from_results(correlation_df, electrodes)

if __name__ == "__main__":
    main()
