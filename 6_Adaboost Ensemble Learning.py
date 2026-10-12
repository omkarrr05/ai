# Practical 6 - AdaBoost Ensemble Learning
# Using Wine Dataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Load Wine dataset (178 samples, 13 features, 3 wine classes)
data = load_wine()
X = data.data
y = data.target

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Individual weak classifier (decision stump = tree with only 1 split)
weak = DecisionTreeClassifier(max_depth=1, random_state=42)
weak.fit(X_train, y_train)
weak_accuracy = weak.score(X_test, y_test)

# AdaBoost ensemble of 50 weak classifiers
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
ada.fit(X_train, y_train)
ada_accuracy = ada.score(X_test, y_test)

# Compare both models
print("Weak Classifier Accuracy:", round(weak_accuracy * 100, 2), "%")
print("AdaBoost Accuracy:", round(ada_accuracy * 100, 2), "%")

# Predict a new wine
wine = [X_test[0]]
prediction = ada.predict(wine)
print("Predicted Wine:", data.target_names[prediction[0]])
