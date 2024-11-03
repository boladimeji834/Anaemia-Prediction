# train_model.py

import joblib
from sklearn.ensemble import RandomForestClassifier
from data_preprocessing import load_data, preprocess_data, split_data

def train_model(X_train, y_train):
    """Train a RandomForest model on the training data."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# logic to activate the functions only if they are called in the main progrem

if __name__ == "__main__":
    filepath = "datasets/file_.csv"
    df = load_data(filepath)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and save the model
    model = train_model(X_train, y_train)
    joblib.dump(model, "models/anaemia_model.pkl")
    print("Model training completed and saved.")
