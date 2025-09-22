import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_step, init_trainer
import optax

# Generate synthetic sequential data
jax.random.seed(0)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = jnp.linspace(0, 4 * 3.14159, num=num_samples).reshape(-1, 1)
y = jnp.sin(X)

# Prepare data for RNN
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.array(in_seq), jnp.array(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

# Define the RNN Model using Flax
class RNNModel(nn.Module):
    hidden_size: int
    def setup(self):
        self.rnn = nn.RNN(features=self.hidden_size, num_layers=1)
        self.fc = nn.Dense(1)
        self.relu = nn.relu

    def __call__(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the RNN
        return out

# Initialize the model and optimizer
model = RNNModel(hidden_size=50)
optimizer = optax.adam(learning_rate=0.001)
params = model.init(jax.random.PRNGKey(42), jnp.zeros((1, sequence_length, 1)))['params']

# Define the loss function
def loss_fn(params, x):
    predictions = model.apply({'params': params}, x)
    return jnp.mean((predictions - y_seq[jnp.newaxis, ...]) ** 2)

# Training loop
epochs = 500
for epoch in range(epochs):
    for i in range(len(X_seq)):
        x_batch = X_seq[i][jnp.newaxis, ...]  # Add batch dimension
        y_batch = y_seq[i][jnp.newaxis, ...]  # Add batch dimension

        grads = jax.grad(loss_fn)(params, x_batch)
        updates, opt_state = optimizer.update(grads, params, optax.EmptyState())
        params = optax.apply_updates(params, updates)

    loss = loss_fn(params, X_seq[i][jnp.newaxis, ...])
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Testing on new data
X_test = jnp.linspace(4 * 3.14159, 5 * 3.14159, num=10).reshape(-1, 1)
x_test_batch = X_test[jnp.newaxis, ...]  # Add batch dimension

predictions = model.apply({'params': params}, x_test_batch)
print(f"Predictions for new sequence: {predictions.tolist()}")