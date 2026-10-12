# Practical 7 - Naive Bayes' Classifier
# Using Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

# Create and train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Class probabilities (prior probability of each class in training data)
for name, prob in zip(data.target_names, model.class_prior_):
    print("P(" + name + ") =", round(prob, 3))

# Calculate accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", round(accuracy * 100, 2), "%")

# Predict the first test patient with probabilities
patient = X_test[:1]
prediction = model.predict(patient)
probability = model.predict_proba(patient)[0]
print("Predicted Patient:", data.target_names[prediction[0]])
print("Probabilities [malignant, benign]:", probability.round(3))
