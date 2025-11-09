# ===============================================
# 🏦 Bank Customer Churn Prediction using ANN
# ===============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Step 1: Load the dataset
# ------------------------------
data = pd.read_csv(r"c:\Users\Public\Documents\Custom Office Templates\ML\Churn_Modelling.csv")

print("✅ Dataset Loaded Successfully!")
print("Shape of dataset:", data.shape)
print("\nSample Data:\n", data.head())

# ------------------------------
# Step 2: Feature & Target Split
# ------------------------------
# Drop unnecessary columns
X = data.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])
y = data["Exited"]

# Encode categorical features
label_encoder_geo = LabelEncoder()
label_encoder_gen = LabelEncoder()
X["Geography"] = label_encoder_geo.fit_transform(X["Geography"])
X["Gender"] = label_encoder_gen.fit_transform(X["Gender"])

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 3: Normalize the data
# ------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# Step 4: Build the ANN model
# ------------------------------
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# ------------------------------
# Step 5: Model Evaluation
# ------------------------------
# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 6: Visualize Accuracy
# ------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('📈 Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Step 7: Plot Confusion Matrix
# ------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Stayed", "Exited"],
            yticklabels=["Stayed", "Exited"])
plt.title("Confusion Matrix - Bank Customer Churn")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
