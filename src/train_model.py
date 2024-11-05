import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_preprocessing import split_data, preprocess_data
import os, pandas as pd 

# Change the current working directory to the project roo
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
print("Changed working directory to:", os.getcwd())

def create_pipeline():
    """this function create a pipeline with preprocessing and model training."""
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Sex']),  # One-hot encode categorical feature 'Sex'
            ('num', StandardScaler(), ['%Red Pixel', '%Green pixel', '%Blue pixel', 'Hb'])
        ]
    )

    #the pipeline with preprocessing and model
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

    # create the pipeline and train it
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    
    print("pipeline model training completed and saved.")
    print("this is the format of the training data: ")
    print(X_test.head())
    print(y_test.head())

    # save the entire pipeline as a joblib file
    joblib.dump(pipeline, "models/anaemia_pipeline.joblib")
    print("Pipeline model training completed and saved.")
    print("mo")
