import re
import nltk
from bs4 import BeautifulSoup
nltk.download('stopwords')
from nltk.corpus import stopwords


def lowercase(text):
    return text.lower()

# removal of HTML Contents
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

# remove urls
def remove_url(text):
    return re.sub(r'http\S+', '', text)

# remove inbetween brackets
def remove_punctuation(text):
    return re.sub('[^\w\s]', '', text)

# Removal of Special Characters
def remove_numbers(text):
    return re.sub('\d', '', text)

# Removal of stopwords
def remove_stopwords_and_lemmatization(text):
    final_text = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    for word in text:
        if word not in set(stopwords.words('english')):
            lemma = nltk.WordNetLemmatizer()
            word = lemma.lemmatize(word)
            final_text.append(word)
    return final_text

def clean_text(text):
    text = lowercase(text)
    text = remove_html(text)
    text = remove_url(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stopwords_and_lemmatization(text)
    return text