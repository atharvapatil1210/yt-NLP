import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

# "Bag of Words"
# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')
X, y = newsgroups.data, newsgroups.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for vectorizing and classifying
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

# Save the model for deployment
joblib.dump(pipeline, 'news_categorizer.pkl')

# Load the model and make predictions (example of usage)
loaded_model = joblib.load('news_categorizer.pkl')
sample_text = ["The new iPhone 12 features a powerful A14 Bionic chip and 5G capability."]
prediction = loaded_model.predict(sample_text)
print(f"Predicted category: {newsgroups.target_names[prediction[0]]}")
