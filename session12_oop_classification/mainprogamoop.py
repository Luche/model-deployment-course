"""
Session 12 – OOP Classification with GridSearchCV and Pickle
Topics: DataHandler class, ModelHandler class, GridSearchCV hyperparameter tuning,
        pickle model serialization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Data Handler Class
# ─────────────────────────────────────────────────────────────────────────────
class DataHandler:
    def __init__(self, file_path):
        self.file_path  = file_path
        self.data       = None
        self.input_df   = None
        self.output_df  = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df  = self.data.drop(target_column, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Model Handler Class
# ─────────────────────────────────────────────────────────────────────────────
class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data  = input_data
        self.output_data = output_data
        self.x_train = self.x_test = self.y_train = self.y_test = self.y_predict = None
        self.createModel()

    def checkAgeOutlierWithBox(self, kolom):
        boxplot = self.x_train.boxplot(column=[kolom])
        plt.show()

    def createMeanFromColumn(self, kolom):
        return np.mean(self.x_train[kolom])

    def createModel(self, criteria='gini', maxdepth=6):
        self.model = RandomForestClassifier(criterion=criteria, max_depth=maxdepth)

    def dataConvertToNumeric(self, columns):
        self.x_train[columns] = pd.to_numeric(self.x_train[columns], errors='coerce')
        self.x_test[columns]  = pd.to_numeric(self.x_train[columns], errors='coerce')

    def fillingNAWithNumbers(self, columns, number):
        self.x_train[columns] = self.x_train[columns].fillna(number)
        self.x_test[columns]  = self.x_test[columns].fillna(number)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test)

    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict,
                                     target_names=['1', '2', '3', '4', '5', '6']))

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data,
            test_size=test_size, random_state=random_state)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def tuningParameter(self):
        parameters = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [2, 4, 6, 8],
        }
        gs = GridSearchCV(RandomForestClassifier(),
                          param_grid=parameters,
                          scoring='accuracy',
                          cv=5)
        gs.fit(self.x_train, self.y_train)
        print("Best params:   ", gs.best_params_)
        print("Best CV score: ", gs.best_score_)
        self.createModel(criteria=gs.best_params_['criterion'],
                         maxdepth=gs.best_params_['max_depth'])

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dermatology-like dataset (used when real CSV is absent)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_dermatology(n=366, seed=42):
    rng = np.random.default_rng(seed)
    feature_cols = [
        'erythema', 'scaling', 'definite_borders', 'itching', 'koebner_phenomenon',
        'polygonal_papules', 'follicular_papules', 'oral_mucosal_involvement',
        'knee_and_elbow_involvement', 'scalp_involvement', 'family_history',
        'melanin_incontinence', 'eosinophils_infiltrate', 'PNL_infiltrate',
        'fibrosis_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis',
        'parakeratosis', 'clubbing_rete_ridges', 'elongation_rete_ridges',
        'thinning_suprapapillary_epidermis', 'spongiform_pustule', 'munro_microabcess',
        'focal_hypergranulosis', 'disappearance_granular_layer',
        'vacuolisation_damage_basal_layer', 'spongiosis', 'saw_tooth_appearance_retes',
        'follicular_horn_plug', 'perifollicular_parakeratosis',
        'inflammatory_mononuclear_infiltrate', 'band_like_infiltrate',
    ]
    data = {col: rng.integers(0, 4, n) for col in feature_cols}
    # 'age' column: mostly numeric strings, occasional '?' to mimic real data
    ages = rng.integers(10, 80, n).astype(str).tolist()
    for i in rng.choice(n, size=int(0.02 * n), replace=False):
        ages[i] = '?'
    data['age'] = ages
    data['class'] = rng.choice([1, 2, 3, 4, 5, 6], n,
                                 p=[0.31, 0.17, 0.20, 0.13, 0.14, 0.05])
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────
file_path = Path(__file__).parent / 'dermatology_database_1.csv'

if not file_path.exists():
    print("dermatology_database_1.csv not found – generating synthetic data...")
    synthetic_df = generate_synthetic_dermatology()
    synthetic_df.to_csv(file_path, index=False)
    print(f"Synthetic dataset saved to {file_path}")

data_handler = DataHandler(str(file_path))
data_handler.load_data()
data_handler.create_input_output('class')

input_df  = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()
model_handler.dataConvertToNumeric('age')

age_mean = model_handler.createMeanFromColumn('age')
model_handler.fillingNAWithNumbers('age', age_mean)

print("── Before Parameter Tuning ─────────────────")
model_handler.train_model()
print(f"Model Accuracy: {model_handler.evaluate_model():.4f}")
model_handler.makePrediction()
model_handler.createReport()

print("\n── After Parameter Tuning (GridSearchCV) ───")
model_handler.tuningParameter()
model_handler.train_model()
print(f"Model Accuracy: {model_handler.evaluate_model():.4f}")
model_handler.makePrediction()
model_handler.createReport()

model_handler.save_model_to_file('trained_model.pkl')
print("\nSession 12 completed successfully!")
