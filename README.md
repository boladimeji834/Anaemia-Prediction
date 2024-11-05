# Anemia Prediction Project

Anemia is a common health condition that, if left untreated, can lead to serious complications. This project uses machine learning to predict the likelihood of anemia in patients based on various health features, supporting early diagnosis and timely intervention.

![Project Preview](images/project_preview.jpg)

## Table of Contents
- ![Project Preview](../images/project_preview.jpg)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project employs machine learning algorithms to predict anemia risk using patient health metrics. With the model's predictions, healthcare providers can identify patients at risk for anemia and take preemptive steps to improve health outcomes.

## Dataset
The dataset includes anonymized patient health information such as:
- Hemoglobin levels
- Red blood cell counts
- Hematocrit values
- Age
- Additional biomarkers and demographic features

*(Specify the source of your dataset, or if it’s synthetic, mention that as well)*

## Technologies Used
- **Python**
- **Pandas** and **NumPy** for data processing
- **Scikit-learn** for machine learning
- **Matplotlib** and **Seaborn** for data visualization
- **Streamlit** for deploying an interactive web interface

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/boladimeji834/Anaemia-Prediction.git
    cd anemia-prediction
    ```

2. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. **Data Preprocessing**: Ensure that the dataset is properly preprocessed. Use the provided scripts to clean and prepare the data.
    ```bash
    python src/data_preprocessing.py
    ```

2. **Model Training**:
    Train the machine learning model with:
    ```bash
    python src/train_model.py
    ```

3. **Evaluate Model**:
    After training, evaluate the model’s performance using:
    ```bash
    python src/evaluate_model.py
    ```

4. **Prediction**:
   Launch the app (using Streamlit):
    ```bash
    streamlit run src/app.py
    ```
   Open your browser at `http://localhost:8501` to access the interface.

## Model Performance
The model achieved an accuracy of **X%** on the test set. Key metrics include:
- **Accuracy**: 99%
- **Precision**: 97%
- **Recall**: 100%
- **F1 Score**: 99%

## Project Structure
anemia-prediction/
├── data/               
├── notebooks/           
├── src/                 
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── app.py           
└── requirements.txt              

