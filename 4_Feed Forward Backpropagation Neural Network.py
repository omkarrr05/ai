# Practical 4 - Feed Forward Backpropagation Neural Network
# Using Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load Breast Cancer dataset (569 patients, 30 features)
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

# Create Neural Network model (30 -> 16 -> 8 -> 1)
model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    max_iter=500,
    random_state=42
)

# Train the model (feed forward + backpropagation)
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Predict the first test patient
patient = X_test[:1]
prediction = model.predict(patient)
print("Predicted Patient:", data.target_names[prediction[0]])
print("Actual Patient:", data.target_names[y_test[0]])
