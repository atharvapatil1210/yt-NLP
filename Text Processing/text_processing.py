import nltk
import spacy
from nltk.tokenize import word_tokenize #Tokenization
from nltk.stem import PorterStemmer #Stemming
from spacy.lang.en import English # Lemmatization

# Download necessary NLTK data
nltk.download('punkt')

# Initialize the stemmer
stemmer = PorterStemmer()

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Natural Language Processing is fascinating. Running, runner, and ran are forms of the word run. Better is the comparative form of good."

# Tokenization using NLTK
tokens = word_tokenize(text)
print("Tokenization:")
print(tokens)

# Stemming using NLTK
stems = [stemmer.stem(token) for token in tokens]
print("\nStemming:")
print(stems)

# Lemmatization using spaCy
doc = nlp(text)
lemmas = [token.lemma_ for token in doc]
print("\nLemmatization:")
print(lemmas)

