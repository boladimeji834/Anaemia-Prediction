import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column="Anaemic"):
    """Separate features and target without scaling."""
    X = df.drop(columns=[target_column])
    y = df[target_column].map({"Yes": 1, "No": 0})
    X["Sex"] = X["Sex"].str.strip()

    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Main script
if __name__ == "__main__":
    

    # Preprocess the data
    target_column = 'Anaemic'  # Update this to match your dataset's target column name
    X, y = preprocess_data(
        df = pd.read_csv("../datasets/file_.csv").set_index("Number"))
    
    # X["Sex"] = X["Sex"].str.strip()

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Data split successfully!")
