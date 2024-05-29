### Practical Example: Text Classification Using Different Methods
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import TruncatedSVD
import gensim
from gensim.models import Word2Vec

# Load the dataset
newsgroups = fetch_20newsgroups(subset='all')
X, y = newsgroups.data, newsgroups.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Bag of Words Model
vectorizer_bow = CountVectorizer(stop_words='english')
X_train_bow = vectorizer_bow.fit_transform(X_train)
X_test_bow = vectorizer_bow.transform(X_test)

# Train and evaluate the Bag of Words model
classifier_bow = MultinomialNB()
classifier_bow.fit(X_train_bow, y_train)
y_pred_bow = classifier_bow.predict(X_test_bow)

print("Bag of Words Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_bow) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred_bow, target_names=newsgroups.target_names))

# 2. TF-IDF Model
vectorizer_tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# Train and evaluate the TF-IDF model
classifier_tfidf = MultinomialNB()
classifier_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = classifier_tfidf.predict(X_test_tfidf)

print("TF-IDF Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred_tfidf, target_names=newsgroups.target_names))

# 3. Word2Vec Model
def preprocess_text(text):
    # Tokenize and clean text
    return gensim.utils.simple_preprocess(text)

X_train_tokens = [preprocess_text(doc) for doc in X_train]
X_test_tokens = [preprocess_text(doc) for doc in X_test]

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=X_train_tokens, vector_size=100, window=5, min_count=2, workers=4)
word2vec_model.train(X_train_tokens, total_examples=len(X_train_tokens), epochs=10)

def document_vector(doc):
    # Create document vector by averaging word vectors
    doc = [word for word in doc if word in word2vec_model.wv.index_to_key]
    return np.mean(word2vec_model.wv[doc], axis=0) if len(doc) > 0 else np.zeros(100)

X_train_w2v = np.array([document_vector(doc) for doc in X_train_tokens])
X_test_w2v = np.array([document_vector(doc) for doc in X_test_tokens])

# Train and evaluate the Word2Vec model
classifier_w2v = MultinomialNB()
classifier_w2v.fit(X_train_w2v, y_train)
y_pred_w2v = classifier_w2v.predict(X_test_w2v)

print("Word2Vec Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred_w2v) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred_w2v, target_names=newsgroups.target_names))