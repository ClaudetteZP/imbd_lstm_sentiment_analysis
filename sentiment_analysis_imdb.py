import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load dataset
num_words = 10000  # Keep only the top 10,000 words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences
max_length = 200
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Build the LSTM model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=num_words, output_dim=64, input_length=max_length),
    keras.layers.LSTM(64, return_sequences=True),
    keras.layers.LSTM(32),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Function to predict sentiment of a review
def predict_sentiment(review):
    encoded_review = [imdb.get_word_index().get(word, 0) for word in review.split()]
    padded_review = pad_sequences([encoded_review], maxlen=max_length)
    prediction = model.predict(padded_review)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment

# Example reviews
reviews = [
    "This movie was absolutely fantastic, I loved every moment!",
    "Terrible film, I regret watching it.",
    "An average movie, nothing special but not bad.",
    "One of the best performances I've ever seen!",
    "Boring and predictable story."
]

# Predict sentiment for each review
for review in reviews:
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {predict_sentiment(review)}")