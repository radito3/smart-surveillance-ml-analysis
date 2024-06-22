import tensorflow as tf
import tensorflow_hub as hub

# https://www.tensorflow.org/hub/tutorials/movinet
# https://www.kaggle.com/models/google/movinet/tensorFlow2/a2-stream-kinetics-600-classification

hub_url = "https://www.kaggle.com/models/google/movinet/TensorFlow2/a2-stream-kinetics-600-classification/2"

encoder = hub.KerasLayer(hub_url, trainable=True)

# Define the image (video) input
image_input = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

# Define the state inputs, which is a dict that maps state names to tensors.
init_states_fn = encoder.resolved_object.signatures['init_states']
state_shapes = {
    name: ([s if s > 0 else None for s in state.shape], state.dtype)
    for name, state in init_states_fn(tf.constant([0, 0, 0, 0, 3])).items()
}
states_input = {
    name: tf.keras.Input(shape[1:], dtype=dtype, name=name)
    for name, (shape, dtype) in state_shapes.items()
}

# The inputs to the model are the states and the video
inputs = {**states_input, 'image': image_input}

outputs = encoder(inputs)

model = tf.keras.Model(inputs, outputs, name='movinet')

# Create your example input here.
# Refer to the description or paper for recommended input shapes.
example_input = tf.ones([1, 8, 172, 172, 3])

# Split the video into individual frames.
# Note: we can also split into larger clips as well (e.g., 8-frame clips).
# Running on larger clips will slightly reduce latency overhead, but
# will consume more memory.
frames = tf.split(example_input, example_input.shape[1], axis=1)

# Initialize the dict of states. All state tensors are initially zeros.
init_states = init_states_fn(tf.shape(example_input))

# Run the model prediction by looping over each frame.
states = init_states
predictions = []
for frame in frames:
    output, states = model({**states, 'image': frame})
    predictions.append(output)

# The video classification will simply be the last output of the model.
final_prediction = tf.argmax(predictions[-1], -1)

# Alternatively, we can run the network on the entire input video.
# The output should be effectively the same
# (but it may differ a small amount due to floating point errors).
non_streaming_output, _ = model({**init_states, 'image': example_input})
non_streaming_prediction = tf.argmax(non_streaming_output, -1)
