from keras.models import Sequential
from keras.layers import LSTM, Dense

# read vector from FIFO, pass through LSTM NN if needed and send notification if sus behaviour detected


# Assuming each of our input sequences has a dimensionality of 'n_features'
n_features = 10
n_classes = 3

# Assuming your feature vectors have been transposed into the shape
# (n_samples, timesteps, n_features)

# Initialize a Sequential model
model = Sequential()

# Add two LSTM layers with dropout
model.add(LSTM(32, return_sequences=True, input_shape=(None, n_features)))
# return_sequences=True means the LSTM layer will return the full sequence, not just the output at the last timestep
model.add(LSTM(32, return_sequences=True))

# Add a Dense layer to output predictions
model.add(Dense(n_classes, activation='softmax'))

# Compile the model with appropriate loss function, optimizer and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()
