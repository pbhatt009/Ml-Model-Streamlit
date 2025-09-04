
# # text preprocessor
from sklearn.base import BaseEstimator, TransformerMixin,ClassifierMixin
import re
import string
# import nltk

# from nltk.corpus import stopwords
import spacy
import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

#     # Download stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# Load spacy model


# chat_words = {
#     "wtf": "what the hell",
#     "omg": "oh my god",
#     "ffs": "for god's sake",
#     "smh": "shaking my head",
#     "fml": "hate my life",
#     "ugh": "annoyed",
#     "ripoff": "overpriced or unfair",
#     "scam": "fraud",
#     "bruh": "are you serious",
#     "lame": "boring or disappointing",
#     "crap": "bad quality",
#     "damn": "annoyed",
#     "bs": "nonsense",
#     "hell": "frustrated",
#     "worst": "very bad",
#     "wt": "what",
#     "af": "very",
#     "idc": "i do not care",
#     "btw": "by the way",
#     "u": "you",
#     "ur": "your",
#     "urs": "yours",
#     "r": "are",
#     "k": "okay",
#     "ok": "okay",
#     "pls": "please",
#     "plz": "please",
#     "thx": "thanks",
#     "tx": "thanks",
#     "ty": "thank you",
#     "yw": "you're welcome",
#     "idk": "I don't know",
#     "imo": "in my opinion",
#     "b4": "before",
#     "gr8": "great",
#     "l8r": "later",
#     "lol": "laugh out loud",
#     "rofl": "rolling on the floor laughing",
#     "lmao": "laughing my ass off",
#     "np": "no problem",
#     "tbh": "to be honest",
#     "bff": "best friends forever",
#     "msg": "message",
#     "txt": "text",
#     "cuz": "because",
#     "bc": "because",
#     "bday": "birthday",
#     "afaik": "as far as I know",
#     "asap": "as soon as possible",
#     "fyi": "for your information",
#     "nvm": "never mind",
#     "tho": "though",
#     "gf": "girlfriend",
#     "bf": "boyfriend",
#     "ttyl": "talk to you later",
#     "omw": "on my way",
#     "ily": "I love you",
#     "ikr": "I know right",
#     "yolo": "you only live once",
#     "wth": "what the hell",
#     "sup": "what's up",
#     "coz": "because",
#     "cus": "because"
# }


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, chat_words):
        self.chat_words = chat_words
      

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.preprocess_text(text) for text in X]

    def preprocess_text(self, text):
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # 3. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 4. Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # 5. Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # 6. Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 7. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 8. Remove XXXX-like placeholders
        text = re.sub(r'\b[xX]{2,}\b', '', text)
        
        # 9. Replace chat words
        words = text.split()
        words = [self.chat_words.get(w, w) for w in words]
        text = " ".join(words)
        
        # 10. Tokenization + Stopword removal + Lemmatization
        doc = nlp(text)
        tokens=[]
        for token in doc:
        
             # Skip stopwords like "is", "the", "and"
            if token.is_stop:
                continue
        
            # Skip spaces or empty strings
            if token.is_space:
                continue
        
            # Add the lemma (base form) of the token to our list
            tokens.append(token.lemma_)
        
        return " ".join(tokens)

