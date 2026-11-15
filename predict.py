
# ==========================================
# IRIS FLOWER PREDICTION
# ==========================================

# Import joblib
import joblib

# Load the saved model
model = joblib.load("iris_model.pkl")

# Enter flower measurements
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

# Prepare input for prediction
flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Predict flower species
prediction = model.predict(flower)

# Display prediction
print("\nPredicted Flower Species:", prediction[0])
