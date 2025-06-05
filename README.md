Tweet Sentiment Analysis with GRU

This project implements a modular pipeline for tweet sentiment analysis using a Gated Recurrent Unit (GRU) model built with PyTorch. It processes tweet data, cleans and preprocesses text, trains a GRU-based classifier, evaluates its performance, and provides predictions for new tweets. The pipeline is designed for extensibility and includes logging for debugging.

Project Overview

The pipeline performs sentiment analysis on tweets, classifying them as positive, neutral, or negative. Key features:


Data Preprocessing: Cleans tweets using NLTK (lowercasing, lemmatization, stopword removal) and converts text to word indices for GRU input.


Model: Uses a GRU neural network with embedding, GRU, and fully connected layers, trained with PyTorch.


Evaluation: Computes loss, accuracy, classification report, and confusion matrix.


Logging: Each module logs progress and errors to separate files in the logs/ directory.
