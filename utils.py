import re
from textblob import TextBlob
from nltk.corpus import stopwords


def singularize_sentence(txt):
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


def remove_punctuation(txt):
    """Remove all punctuation from the text."""
    txt = re.sub(r'[?|!|\'|"|#]', r'', txt)
    txt = re.sub(r'[.|,|)|(|\|/]', r' ', txt)
    txt = txt.strip()
    txt = txt.replace('\n', ' ')
    return txt


def preprocess(txt, stop=None, stem=None, punct=None, single=None):
    """Preprocess the input text according to the user's input."""
    if punct:
        txt = remove_punctuation(txt)
    if stop:
        txt = remove_stopwords(txt)
    if single:
        txt = singularize_sentence(txt)
    if stem:
        sentence = stem_text(txt)
    return txt


def predict(txt):
    return txt
