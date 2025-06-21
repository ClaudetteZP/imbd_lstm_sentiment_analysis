# IMDB Sentiment Analysis Using TensorFlow and Keras

This repository demonstrates a sentiment analysis pipeline built with TensorFlow and Keras. It uses LSTM neural networks to classify IMDB movie reviews as positive or negative.

## Overview

- Utilizes the IMDB dataset provided by Keras  
- Applies sequence padding and word embeddings  
- Implements a stacked LSTM architecture for deep sequence learning  
- Provides a custom function to predict sentiment from user input

## Model Architecture

- **Embedding Layer**: Converts word indices to dense vectors  
- **LSTM (64 units)**: Returns full sequences for contextual learning  
- **LSTM (32 units)**: Processes sequential output  
- **Dense Layer**: Outputs binary prediction using sigmoid activation


