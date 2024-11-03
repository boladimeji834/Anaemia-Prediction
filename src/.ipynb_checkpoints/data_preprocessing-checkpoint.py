import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib  # For saving the processed data

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath).set_index("Number")

def preprocess_data(df):
    """Preprocess the dataset by handling missing values, encoding categorical variables, and scaling features."""
    X = df.drop(columns=['Anaemic'])  # Target variable
    y = df['Anaemic'].map({"Yes": 1, "No": 0})
    
    # Define numeric and categorical transformations
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MaxAbsScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X = preprocessor.fit_transform(X)

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def save_processed_data(X_train, X_test, y_train, y_test, output_dir="datasets/"):
    """Save preprocessed and split data to disk."""
    joblib.dump((X_train, X_test, y_train, y_test), f"{output_dir}/processed_data.joblib")
    print(f"Data saved to '{output_dir}/processed_data.joblib'.")

if __name__ == "__main__":
    filepath = filepath = "C:/Users/eigen/Desktop/Anaemia Prediction/datasets/file_.csv"

    df = load_data(filepath)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test)
    print("Data preprocessed, split, and saved successfully.")
