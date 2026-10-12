# Practical 3 - Decision Tree Learning
# Using Iris Dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load Iris dataset
iris = load_iris()

# Take first 10 records
X = iris.data[11:60]
#print("x: ",X)
y = iris.target[11:60]
#print("y: ",y)

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)

# Create Decision Tree model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy * 100, "%")

# Give a new flower for prediction
flower = [[6.9, 3.1, 4.9, 1.5]]

# Predict the flower type
prediction = model.predict(flower)

print("Predicted Flower:", iris.target_names[prediction[0]])
