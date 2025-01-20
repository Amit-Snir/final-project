import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter

def determine_emotional_state(tp9_outliers, tp10_outliers, af8_outliers, af7_outliers, wave):
    #Determine emotional state based on outliers and wave type.
    state = None

    if wave == 'ALPHA':
        if tp9_outliers > 7 or tp10_outliers > 5:
            state = 'R'
        elif af8_outliers > 1:
            state = 'N'
        elif af7_outliers > 1:
            state = 'C'
        elif 0.6 <= af7_outliers <= 1:
            state = 'N'
        else:
            state = 'R'

    elif wave == 'BETA':
        if af8_outliers > 0.35:
            state = 'C'

    elif wave == 'DELTA':
        if all(x < 180 for x in [tp9_outliers, tp10_outliers, af8_outliers, af7_outliers]):
            state = 'C'

    elif wave == 'GAMMA':
        if tp9_outliers > 0.1:
            state = 'R'
        elif af8_outliers > 0.85:
            state = 'C'

    elif wave == 'THETA':
        if all(x > 150 for x in [tp9_outliers, tp10_outliers, af8_outliers, af7_outliers]):
            state = 'R'

    return state

def load_and_prepare_data(file_path):
    """Load and prepare the dataset with simplified logic."""
    try:
        data = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        print(f"Error loading file: {file_path}. Error: {e}")
        return None

    try:
        data['Emotional_State'] = data.apply(
            lambda row: determine_emotional_state(
                tp9_outliers=row['TP9'],
                tp10_outliers=row['TP10'],
                af8_outliers=row['AF8'],
                af7_outliers=row['AF7'],
                wave=row['wave']
            ), axis=1
        )

        # Remove rows where Emotional_State is None
        data = data.dropna(subset=['Emotional_State'])

    except Exception as e:
        print(f"Error processing data in file: {file_path}. Error: {e}")
        return None

    return data

def train_random_forest(train_data):
    #Train a Random Forest Classifier on the prepared data.
    try:
        # Select features and target
        feature_columns = ['TP9', 'TP10', 'AF8', 'AF7']
        X = train_data[feature_columns]
        y = train_data['Emotional_State']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest Classifier
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Print evaluation metrics
        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Print feature importance
        feature_importances = model.feature_importances_
        print("Feature Importances:")
        for feature, importance in zip(feature_columns, feature_importances):
            print(f"{feature}: {importance}")

        return model
    except Exception as e:
        print(f"Error during model training or evaluation. Error: {e}")
        return None

def plot_score_distribution(scores):
    #Plot the score distribution for the participant.
    try:
        scores.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Score Distribution for Participant')
        plt.xlabel('Emotional Category', rotation=0)
        plt.xticks(rotation=0)
        plt.ylabel('Score')
        plt.grid(axis='y')
        plt.show()
    except Exception as e:
        print(f"Error during plotting. Error: {e}")

def predict_emotional_state(model, participant_file):
    #Predict emotional state for a participant using the trained model.
    try:
        participant_data = load_and_prepare_data(participant_file)
        if participant_data is None:
            print("Failed to load participant data.")
            return None

        feature_columns = ['TP9', 'TP10', 'AF8', 'AF7']
        X_participant = participant_data[feature_columns]
        predictions = model.predict(X_participant)

        # Display the predicted state
        # Count occurrences of each emotional state
        state_counts = Counter(predictions)
        print(f"Emotional State Counts: {state_counts}")

        # Plot the counts of emotional states
        state_series = pd.Series(state_counts)
        plot_score_distribution(state_series)
        return predictions
    except Exception as e:
        print(f"Error during prediction. Error: {e}")
        return None

def main():
    #Main execution function.
    train_file_path = r"C:\python advenced\final-project\data\4. data_ready_for_analysis\machine_training_data.xlsx"
    participant_file_path = r"C:\python advenced\final-project\data\4. data_ready_for_analysis\ready_for_analysis.xlsx"

    # Step 1: Load and prepare the training data
    train_data = load_and_prepare_data(train_file_path)
    if train_data is None:
        print("Failed to prepare training data. Exiting.")
        return

    # Step 2: Train Random Forest Classifier
    model = train_random_forest(train_data)
    if model is None:
        print("Model training failed. Exiting.")
        return

    # Step 3: Predict emotional state for a participant
    predict_emotional_state(model, participant_file_path)

if __name__ == "__main__":
    main()
