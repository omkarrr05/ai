# Practical 5 - Support Vector Machine (SVM)
# Using Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load Breast Cancer dataset (569 patients, 30 features, 2 classes)
data = load_breast_cancer()
X = data.data
y = data.target

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale the features (fit on training data only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train SVM model
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Optimize parameters (C, gamma, kernel) using GridSearchCV
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.1],
    "kernel": ["rbf", "linear"]
}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# Accuracy of the optimized model
best_accuracy = grid.score(X_test, y_test)
print("Optimized Accuracy:", round(best_accuracy * 100, 2), "%")

# Predict the first test patient
prediction = grid.predict(X_test[:1])
print("Predicted Patient:", data.target_names[prediction[0]])
