import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter
import logging
import os


class DataPreprocessor:
    def __init__(self, log_dir="logs"):
        self.logger = logging.getLogger("DataPreprocessor")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            filename=os.path.join(log_dir, "data_preprocessing.log"),
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.word_to_index = None
        self.max_len = None

    def clean_text(self, text):
        try:
            text = text.lower()
            text = re.sub(r"[^a-z\s]", "", text)
            text = [self.lemmatizer.lemmatize(token) for token in text.split() if token not in self.stop_words]
            return " ".join(text).strip()
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            raise

    def preprocess(self, data):
        try:
            data = data.copy()
            data["tweet"] = data["tweet"].apply(self.clean_text)
            data = data[data["tweet"].str.len() > 0].reset_index(drop=True)
            sentiment_mapping = {"positive": 0, "neutral": 1, "negative": 2}
            data["sentiment"] = data["sentiment"].map(sentiment_mapping)
            self.logger.info("Text cleaning and sentiment mapping completed")

            # Create vocabulary
            words = [word for tweet in data["tweet"] for word in tweet.split()]
            words_frequency = Counter(words)
            self.word_to_index = {word: index + 2 for index, (word, _) in enumerate(words_frequency.items())}
            self.word_to_index["<PAD>"] = 0
            self.word_to_index["<UNK>"] = 1

            # Calculate max length
            self.max_len = max(len(tweet.split()) for tweet in data["tweet"])
            self.logger.info(f"Vocabulary size: {len(self.word_to_index)}, Max length: {self.max_len}")
            return data
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def text_to_indices(self, text):
        try:
            tokens = text.split()
            indices = [self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in tokens]
            indices += [self.word_to_index["<PAD>"]] * (self.max_len - len(indices))
            return indices[:self.max_len]
        except Exception as e:
            self.logger.error(f"Error converting text to indices: {str(e)}")
            raise