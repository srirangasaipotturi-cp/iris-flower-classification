
# ==========================================
# PROJECT: IRIS FLOWER CLASSIFICATION
# LESSON 5: LOADING AND UNDERSTANDING DATA
# ==========================================

# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Convert the dataset into a Pandas DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

# Add the flower species column
df["species"] = [
    iris.target_names[target]
    for target in iris.target
]
# First EDA exercise: Count the flowers

print("\nTotal number of flowers:")
print(len(df))
# Display dataset information
print("\n========== IRIS DATASET ==========")

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nFlower species:")
print(iris.target_names)

print("\nNumber of flowers in each species:")
print(df["species"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())
# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# 1. Display dataset information

print("\n========== DATASET INFORMATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSpecies distribution:")
print(df["species"].value_counts())


# ==========================================
# EDA GRAPH 1: SPECIES DISTRIBUTION
# ==========================================

import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folder if necessary
os.makedirs("outputs", exist_ok=True)

# Count the number of flowers in each species
species_counts = df["species"].value_counts()

print("\nFlower species counts:")
print(species_counts)

# Create the bar graph
plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="species"
)

plt.title("Distribution of Iris Flower Species")
plt.xlabel("Flower Species")
plt.ylabel("Number of Flowers")

# Save the graph
plt.savefig("outputs/species_count.png", dpi=300)

# Display the graph
plt.show()


# ==========================================
# EDA GRAPH 2: PETAL LENGTH DISTRIBUTION
# ==========================================

# Create a graph area
plt.figure(figsize=(8, 5))

# Create histogram
sns.histplot(
    data=df,
    x="petal length (cm)",
    hue="species",
    bins=20,
    kde=False
)

# Add title and labels
plt.title("Petal Length Distribution by Species")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Number of Flowers")

# Save the graph
plt.savefig("outputs/petal_distribution.png", dpi=300)

# Display the graph
plt.show()


# ==========================================
# EDA GRAPH 3: PAIRPLOT
# ==========================================

# Create pairplot
pair_plot = sns.pairplot(
    df,
    hue="species"
)

# Add a title
pair_plot.figure.suptitle(
    "Pairplot of Iris Flower Dataset",
    y=1.02
)

# Save pairplot
pair_plot.savefig(
    "outputs/pairplot.png",
    dpi=300
)

# Display graph
plt.show()

# ==========================================
# EDA FINDINGS
# ==========================================

# 1. The dataset contains three Iris species.
# 2. Petal length and petal width help distinguish species.
# 3. Pairplots help visualize relationships between features.


# ==========================================
# MACHINE LEARNING: FEATURES AND TARGET
# ==========================================

# Select the four input features
X = df.drop("species", axis=1)

# Select the target column
y = df["species"]

# Display the features
print("\nFeatures (X):")
print(X.head())

# Display the target
print("\nTarget (y):")
print(y.head())


# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

# Divide the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the dataset sizes
print("\nTraining and Testing Data:")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ==========================================
# MODEL TRAINING
# ==========================================

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=200)

# Train the model using training data
model.fit(X_train, y_train)

# Display confirmation
print("\nModel training completed successfully!")


# ==========================================
# MODEL PREDICTIONS
# ==========================================

# Predict species using testing data
y_pred = model.predict(X_test)

# Display actual and predicted values
print("\nActual Species:")
print(y_test.head(10).values)

print("\nPredicted Species:")
print(y_pred[:10])


# ==========================================
# MODEL EVALUATION
# ==========================================

# Import accuracy score
from sklearn.metrics import accuracy_score

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy
print("\nModel Accuracy:")
print(accuracy * 100, "%")


# ==========================================
# CONFUSION MATRIX
# ==========================================

# Import confusion matrix and plotting tools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(cmap="Blues")

# Add title
plt.title("Confusion Matrix - Iris Classification")

# Save graph
plt.savefig("outputs/confusion_matrix.png", dpi=300)

# Show graph
plt.show()


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

# Import classification report
from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(y_test, y_pred)

# Display report
print("\nClassification Report:")
print(report)


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

# Import joblib
import joblib

# Save trained model
joblib.dump(model, "iris_model.pkl")

print("\nModel saved successfully!")
