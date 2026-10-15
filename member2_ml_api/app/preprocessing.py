"""
preprocessing.py — NLP Preprocessing using NLTK and Regex.
Provides standard text cleaning, tokenization, stopword removal, and lemmatization.
"""

import re
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Handle SSL certificates on Windows for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

# Ensure required NLTK resources are available
for resource in ["stopwords", "punkt", "punkt_tab", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource not in ["punkt", "punkt_tab"] else f"tokenizers/{resource}")
    except (LookupError, OSError):
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    # Fallback common stopwords if NLTK data download fails
    STOP_WORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
        "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
        "t", "can", "will", "just", "don", "should", "now"
    }

try:
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    LEMMATIZER = None


def clean_text(text: str) -> str:
    """
    NLP Preprocessing Pipeline:
    1. Lowercase conversion
    2. Remove URLs, punctuation, special chars, numbers
    3. Tokenize words
    4. Remove NLTK stopwords
    5. Lemmatize tokens
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize by whitespace
    tokens = text.split()

    # Remove stopwords and short words (< 2 chars), and lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 2:
            if LEMMATIZER:
                try:
                    token = LEMMATIZER.lemmatize(token)
                except Exception:
                    pass
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)
