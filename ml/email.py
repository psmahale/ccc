# ===========================================
# 📧 Email Spam Detection using KNN and SVM
# ===========================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# 🔹 Load the dataset
# =============================
data = pd.read_csv(r"emails.csv")

print("✅ Dataset Loaded Successfully!")
print("Shape of dataset:", data.shape)
print("\nSample data:\n", data.head())

# =============================
# 🔹 Prepare the data
# =============================

# Drop unnecessary columns like 'Email No.' if present
if 'Email No.' in data.columns:
    data = data.drop(columns=['Email No.'])

# Separate features (X) and label (y)
X = data.drop(columns=['Prediction'])
y = data['Prediction']

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 🔹 Train KNN Model
# =============================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# =============================
# 🔹 Train SVM Model
# =============================
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# =============================
# 🔹 Evaluate Models
# =============================
print("\n=== KNN Model Performance ===")
print("Accuracy:", round(accuracy_score(y_test, knn_pred), 3))
print("Classification Report:\n", classification_report(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))

print("\n=== SVM Model Performance ===")
print("Accuracy:", round(accuracy_score(y_test, svm_pred), 3))
print("Classification Report:\n", classification_report(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))


#Run Terminal to run code 
#   python "c:\Users\Public\Documents\Custom Office Templates\ML\email.py"
# This program performs Email Spam Detection using two machine learning algorithms:
# K-Nearest Neighbors (KNN) and Support Vector Machine (SVM). 
# The main goal is to classify whether an incoming email is spam or not based on its features.

# First, the program loads the dataset 'emails.csv' using pandas. 
# This dataset contains various numeric features that represent word frequencies, 
# character frequencies, or other text-based metrics extracted from emails. 
# It also includes a target column named 'Prediction' which indicates 
# whether the email is spam (1) or not spam (0).

# After loading, unnecessary columns like 'Email No.' (if present) are dropped to clean the data. 
# The remaining data is then split into two parts: 
# features (X) which contain the independent variables, 
# and label (y) which contains the dependent variable (spam or not).

# The dataset is divided into training and testing subsets 
# using train_test_split(), with 80% data for training the model 
# and 20% for testing. This ensures that the model can be trained 
# and then evaluated on unseen data for fair accuracy measurement.

# The first algorithm used is K-Nearest Neighbors (KNN). 
# KNN works on the concept of similarity — when a new email is given, 
# it checks the 'k' closest data points (here k=5) 
# from the training dataset and predicts the label 
# based on the majority of those neighbors. 
# The KNN model is trained using knn.fit() and tested using knn.predict().

# The second algorithm is Support Vector Machine (SVM) with a linear kernel. 
# SVM finds the best hyperplane that separates spam and non-spam emails 
# in a multi-dimensional feature space. 
# It tries to maximize the margin between both classes for better classification. 
# The model is trained using svm.fit() and predictions are made using svm.predict().

# After training both models, the program evaluates their performance. 
# For both KNN and SVM, it calculates Accuracy, Classification Report, 
# and Confusion Matrix using sklearn.metrics. 
# The accuracy tells the overall correctness of the model, 
# while the classification report provides detailed metrics like precision, recall, and F1-score. 
# The confusion matrix helps visualize how many emails were correctly and incorrectly 
# classified as spam or not spam.

# Finally, both models’ results are printed and compared. 
# Usually, SVM performs slightly better than KNN for text-based datasets 
# because it handles high-dimensional data efficiently. 
# The code thus demonstrates a clear comparison of two classical 
# supervised learning algorithms for spam detection.
