"""
Session 12 – Load and Use a Pickled Model
Demonstrates how to reload a model saved with pickle and run inference.
Run AFTER mainprogamoop.py (which creates trained_model.pkl).
"""

import pickle
import warnings
warnings.filterwarnings('ignore')


def load_model(filename):
    """Load a trained model from a pickle file."""
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_with_model(model, user_input):
    """Make a prediction using the loaded model."""
    prediction = model.predict([user_input])
    return prediction[0]


def main():
    model_filename = 'trained_model.pkl'
    model = load_model(model_filename)
    print(f"Model loaded from {model_filename}")

    # Example input (34 features for dermatology dataset)
    user_input = [2, 1, 2, 3, 1, 3, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1,
                  2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 3, 2, 0, 0, 2, 3, 26]

    prediction = predict_with_model(model, user_input)
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
