import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams


# nltk.download('stopwords')
# nltk.download('wordnet')

def lowercase_text(text: str) -> str:
    return text.lower() 

# Function to remove URLs
def remove_urls(text: str) -> str:
    return re.sub(r'http\S+', '', text)

def tokenize_text(text: str) -> list:
    tokens = word_tokenize(text)
    return tokens 

def remove_punctuation(tokens: list) -> list:
    return [token for token in tokens if token.isalnum()]

def remove_stopwords(tokens: list) -> list:
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

# Function to apply stemming
def apply_stemming(tokens: list) -> list:
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# Function to apply lemmatization
def apply_lemmatization(tokens: list) -> list:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

# Function to handle contractions
def handle_contractions(tokens: list) -> list:
    # Dictionary of common English contractions
    contractions = {
        "can't": "cannot",
        "won't": "will not",
        # Add more contractions as needed
    }
    return [contractions[token] if token in contractions else token for token in tokens]

# Function to remove numeric characters
def remove_numeric(tokens: list) -> list:
    return [token for token in tokens if not token.isdigit()]

def ngrams_generator(text: str, n: int) -> str:
    # Tokenize the text
    tokens = word_tokenize(lowercase_text(text.strip()))
    # Generate n-grams using nltk.ngrams and join them into strings
    result = [' '.join(ngram) for ngram in ngrams(tokens, n)]
    return result

def common_elements(list1, list2):
    # Find common elements between two lists
    return set(list1) & set(list2)