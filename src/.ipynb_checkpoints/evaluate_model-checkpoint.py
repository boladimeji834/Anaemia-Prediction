# evaluate_model.py

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import preprocess_data, split_data
import pandas as pd 

def evaluate_model(model, X_test, y_test):
    """Evaluate the model using test data and print metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {round(accuracy, 2)}")
    print(f"Precision: {round(precision, 2)}")
    print(f"Recall: {round(recall, 2)}")
    print(f"F1 Score: {round(f1, 2)}")

if __name__ == "__main__":
    X, y = preprocess_data(
        df = pd.read_csv("../datasets/file_.csv").set_index("Number"))
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Load the model
    model = joblib.load("../models/anaemia_pipeline.joblib")
    evaluate_model(model, X_test, y_test)
