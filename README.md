
# Student Success Prediction

A machine learning project that predicts whether a student will pass or fail based on academic performance, study habits, attendance, and sleep hours using Logistic Regression.

## Project Overview

This project demonstrates a basic classification workflow using Python and Scikit-learn. It includes data preprocessing, missing value treatment, categorical encoding, feature scaling, train-test splitting, model training, and evaluation.

## Dataset

The dataset contains student-related information used to predict academic outcomes.

**Features:**
- `StudyHours` — Hours spent studying
- `Attendance` — Student attendance percentage
- `PastScore` — Previous academic score
- `SleepHours` — Average hours of sleep
- `Internet` — Internet availability
- `Passed` — Target variable (Pass/Fail)

**Note:** This is a synthetic dataset created for practice and learning purposes. It does not represent real student records.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Workflow

1. **Data Loading:** Imported the dataset using Pandas.
2. **Data Inspection:** Examined the first rows, dataset dimensions, data types, and missing values.
3. **Missing Value Treatment:** Replaced missing attendance and sleep-hour values with their respective column means.
4. **Encoding:** Converted categorical variables (`Internet` and `Passed`) into numerical values using LabelEncoder.
5. **Feature Scaling:** Applied StandardScaler to standardize numerical features.
6. **Train-Test Split:** Split the dataset into 80% training data and 20% testing data.
7. **Model Training:** Trained a Logistic Regression classification model.
8. **Prediction:** Predicted student pass/fail outcomes on the test dataset.
9. **Evaluation:** Used a classification report and confusion matrix to evaluate model performance.

## Machine Learning Model

- **Algorithm:** Logistic Regression
- **Task:** Binary Classification
- **Target:** `Passed`

The model uses the following input features:
- StudyHours
- Attendance
- PastScore
- SleepHours

## Evaluation Metrics

The model is evaluated using:

- **Precision:** Proportion of predicted positives that are correct.
- **Recall:** Proportion of actual positives correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions across both classes.

## Project Structure

```text
Student-Success-Prediction/
│
├── student_success_dataset.csv
├── student_success_prediction.py
└── README.md
```

## How to Run

**1. Clone the repository**

```bash
git clone YOUR_REPOSITORY_URL
```

**2. Navigate to the project folder**

```bash
cd Student-Success-Prediction
```

**3. Install dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**4. Run the Python script**

```bash
python student_success_prediction.py
```

The script prints the classification report and displays the confusion matrix.

## Learning Outcomes

- Data cleaning and preprocessing
- Handling missing values
- Label encoding categorical variables
- Feature standardization
- Training and testing a classification model
- Evaluating model performance
- Visualizing classification results

## Disclaimer

This project is intended for educational purposes only. The dataset is synthetic, and the model's predictions should not be used to make real-world decisions about students.
