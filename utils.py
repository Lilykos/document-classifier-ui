import re
from textblob import TextBlob
from nltk.corpus import stopwords
from classifier import classify


def singularize_words(txt):
    """Singularize words using Textblob, to reduce tokens."""
    tb = TextBlob(txt)
    return ' '.join([word.singularize() for word in tb.words])


def remove_stopwords(txt):
    """Remove stopwords words using Textblob and NLTK."""
    tb = TextBlob(txt)
    return ' '.join([
        word for word in tb.words
        if word not in stopwords.words('english')
    ])


def stem_text(txt):
    """Stem words using Textblob."""
    tb = TextBlob(txt)
    return ' '.join([word.stem() for word in tb.words])


def remove_punctuation_and_numbers(txt):
    """Remove all punctuation from the text."""
    txt = re.sub(r'[,.;@#?!&$]+\ *', ' ', txt)
    txt = re.sub(r'[0 - 9]', r' ', txt)
    txt = txt.strip().replace('\n', ' ')
    return txt


def preprocess(txt, stop=None, stem=None, punct=None, single=None):
    """Preprocess the input text according to the user's input."""
    if punct:
        txt = remove_punctuation_and_numbers(txt)
    if stop:
        txt = remove_stopwords(txt)
    if single:
        txt = singularize_words(txt)
    if stem:
        txt = stem_text(txt)
    return txt


def predict(txt):
    return classify(txt)
