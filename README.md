# Iris Flower Classification 🌸

A Machine Learning project that predicts the species of an Iris flower based on its sepal and petal measurements using Logistic Regression.

## Project Overview

This project uses the famous Iris dataset to classify flowers into three species:

- Setosa
- Versicolor
- Virginica

The model takes four measurements as input and predicts the flower species.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Dataset Features

The model uses four input features:

1. Sepal length (cm)
2. Sepal width (cm)
3. Petal length (cm)
4. Petal width (cm)

**Target:** Iris flower species.

## Project Workflow

1. Load the Iris dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Visualize the dataset using graphs.
4. Separate features and target.
5. Split the dataset into training and testing sets.
6. Train a Logistic Regression model.
7. Make predictions.
8. Evaluate model performance.
9. Save the trained model.
10. Predict new flower species using user input.

## Model Performance

- Algorithm: Logistic Regression
- Training data: 80%
- Testing data: 20%
- Accuracy: **96.67%**

The model was also evaluated using a confusion matrix and classification report.

## Project Structure

```text
iris-flower-classification/
│
├── outputs/
│   ├── species_count.png
│   ├── petal_distribution.png
│   ├── pairplot.png
│   └── confusion_matrix.png
│
├── main.py
├── predict.py
├── iris_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd iris-flower-classification
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### Train the model

```bash
python main.py
```

### Predict a new flower

```bash
python predict.py
```

Enter the four flower measurements when prompted to receive a predicted species.

## Results

The Logistic Regression model achieved **96.67% accuracy** on the test dataset.

## Future Improvements

- Build a web application using Streamlit.
- Compare multiple classification algorithms.
- Improve model evaluation and visualization.
- Deploy the model as a web application.

## Author

AI & Machine Learning Student

## License

This project is intended for educational and learning purposes.