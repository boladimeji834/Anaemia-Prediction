# evaluate_model.py

import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_preprocessing import load_data, preprocess_data, split_data

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
    filepath = "datasets/file_.csv"
    df = load_data(filepath)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Load the model
    model = joblib.load("models/anaemia_model.pkl")
    evaluate_model(model, X_test, y_test)
