IMDB Sentiment Analysis Using Tensorflow and Keras

This repository contains a simple yet effective implementation of a Sentiment Analysis Model trained on the IMDB Movie Reviews
Dataset using TensorFlow and Keras. The model uses LSTM Layers to classify reviews as positive or negative based on their content.

Features
-Preprocessing with Keras' built-in IMDB dataset
-Sequence padding and word embedding
-Stacked LSTM layers for better context capture
-Binary Sentiment Classification with high accuracy
-Custom review prediction function

Model Architecture
-Embedding layer: Maps words to dense vectors
-LSTM (64) layer: Returns sequences
-LSTM (32) layer: Final LSTM output
-Dense layer with sigmoid activation: Outputs binary prediction
