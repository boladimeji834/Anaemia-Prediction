import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_preprocessing import split_data, preprocess_data
import os, pandas as pd 


# Set project root directory dynamically
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Change the current working directory to the project root
os.chdir(project_root)
print("Changed working directory to:", os.getcwd())

def create_pipeline():
    """Create a pipeline with preprocessing and model training."""
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Sex']),  # One-hot encode categorical feature 'Sex'
            ('num', StandardScaler(), ['%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb'])
        ]
    )

    # Define the pipeline with preprocessing and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(project_root, "datasets", "file_.csv")


    X, y = preprocess_data(
        df = pd.read_csv("datasets/file_.csv").set_index("Number"))
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create the pipeline and train it
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Save the entire pipeline as a joblib file
    joblib.dump(pipeline, "models/anaemia_pipeline.joblib")
    print("Pipeline model training completed and saved.")
