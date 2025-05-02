import spacy
import nltk
from nltk.corpus import stopwords, wordnet



class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words("english"))

    def get_simpler_synonym(self, word):
        synsets = wordnet.synsets(word)
        if synsets:
            # Return the first lemma of the first synset
            return synsets[0].lemmas()[0].name()
        else:
            return word

    def extract_keywords(self, text):
        doc = self.nlp(text.lower())
        keywords = []
        for token in doc:
            word = token.text
            if word.isalpha() and word not in self.stop_words:
                simpler_word = self.get_simpler_synonym(word)
                keywords.append(simpler_word)
        return keywords
