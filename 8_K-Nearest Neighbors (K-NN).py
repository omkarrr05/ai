# Practical 8 - K-Nearest Neighbors (K-NN)
# Using Iris Dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

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

# Create and train K-NN model (k = 5 neighbours)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Calculate accuracy
accuracy = knn.score(X_test, y_test)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Predict a new flower
flower = scaler.transform([[5.1, 3.5, 1.4, 0.2]])
prediction = knn.predict(flower)
print("Predicted Flower:", iris.target_names[prediction[0]])
