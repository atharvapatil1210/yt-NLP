from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Sample text data
text_data = [
    "Word embeddings are dense vectors in a continuous vector space.",
    "They capture semantic relationships between words.",
    "Word2Vec is a popular technique for learning word embeddings."
]

# Tokenize the text data
tokenized_data = [word_tokenize(sentence.lower()) for sentence in text_data]

# Train Word2Vec model
model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

# Save the trained model
model.save("word2vec_model.model")

# Load the trained model
loaded_model = Word2Vec.load("word2vec_model.model")

# Access word vectors
word = "word"
print(f"Word vector for '{word}': {loaded_model.wv[word]}")
