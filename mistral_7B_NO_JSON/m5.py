import jax
from jax import numpy as jnp
import flax
from flax import linen as nn
from flax import optim

# Generate synthetic sequential data
rng = jax.random.PRNGKey(42)
sequence_length = 10
num_samples = 100
X = jnp.linspace(0, 4 * 3.14159, num=num_samples).reshape((-1, 1))
y = jnp.sin(X)

# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.stack(in_seq), jnp.stack(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# Define the RNN Model
class RNNModel(nn.Module):
    def setup(self):
        self.rnn = nn.Recurrent(nn.Dense(50), jax.nn.tanh, time_major=True)
        self.fc = nn.Dense(1)

    def __call__(self, carry, inputs):
        carry, outputs = self.rnn.step(carry, inputs)
        return carry, self.fc(outputs[-1, :])  # Use the last output of the RNN

# Initialize the model, loss function, and optimizer
key = jax.random.PRNGKey(0)
rng, dropout_rng = jax.random.split(key)
model = RNNModel()
rng, opt_rng = jax.random.split(dropout_rng)
opt_params = optim.Adam(learning_rate=0.001, use_nesterov=False).init(model.parameters())
criterion = nn.MSELoss()

# Training loop
epochs = 500
for epoch in range(epochs):
    for carry, inputs, target in zip(jax.random.PRNGSequence(opt_rng, jnp.array([1])), X_seq, y_seq):
        with jax.grad():
            outputs = model.apply({"params": opt_params}, (carry, inputs))
            loss = criterion(outputs, target)

        grads = jax.grad(loss, model.parameters())
        opt_updates, opt_params = optim.AdamUpdate(opt_params, grads)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss}")

# Testing on new data
X_test = jnp.linspace(4 * 3.14159, 5 * 3.14159, num=10).reshape((-1, 1))

with jax.random.PRNGKey(key), jax.grad_check(jax.vmap(model.apply)(model.parameters)):
    carry = jnp.zeros((1, model.rnn.state_size))  # Initial RNN state
    predictions = jax.vmap(lambda x: model.apply({"params": opt_params}, (carry, x)))(X_test)
    print(f"Predictions for new sequence: {predictions}")