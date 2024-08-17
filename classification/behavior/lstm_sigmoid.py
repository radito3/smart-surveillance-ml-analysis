import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# `timesteps` and `features` have to be defined based on your input dataset where `timesteps` refers to the sequence
# length and `features` to the number of features each input sequence has

# Data fed into these models should be appropriately preprocessed to fit this input
# shape (`batch_size`, `timesteps`, `features`)

# Define the model architecture
model_single = Sequential([
    LSTM(64, return_sequences=True, input_shape=(timesteps, features)),
    Dropout(0.2),
    LSTM(64),  # experiment with different numbers of neurons (etc. 16, 32)
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_single.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_single.summary()
