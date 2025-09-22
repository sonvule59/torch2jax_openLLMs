import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import matplotlib.pyplot as plt

# Generate synthetic sequential data
jax.random.PRNGKey(42)
sequence_length = 10
num_samples = 100

# Create a sine wave dataset
X = jnp.linspace(0, 4 * 3.14159, num=num_samples).reshape(-1, 1)
y = jnp.sin(X)

# Prepare data for LSTM
def create_in_out_sequences(data, seq_length):
    in_seq = []
    out_seq = []
    for i in range(len(data) - seq_length):
        in_seq.append(data[i:i + seq_length])
        out_seq.append(data[i + seq_length])
    return jnp.array(in_seq), jnp.array(out_seq)

X_seq, y_seq = create_in_out_sequences(y, sequence_length)

class CustomLSTMModel(nn.Module):
    input_dim: int
    hidden_units: int
    
    @nn.compact
    def __call__(self, inputs):
        batch_size, seq_len, _ = inputs.shape
        
        Wxi = self.param('Wxi', jax.random.normal, (self.input_dim, self.hidden_units))
        Whi = self.param('Whi', jax.random.normal, (self.hidden_units, self.hidden_units))
        bi = self.param('bi', jax.random.normal, (self.hidden_units,))
        
        Wxf = self.param('Wxf', jax.random.normal, (self.input_dim, self.hidden_units))
        Whf = self.param('Whf', jax.random.normal, (self.hidden_units, self.hidden_units))
        bf = self.param('bf', jax.random.normal, (self.hidden_units,))
        
        Wxo = self.param('Wxo', jax.random.normal, (self.input_dim, self.hidden_units))
        Who = self.param('Who', jax.random.normal, (self.hidden_units, self.hidden_units))
        bo = self.param('bo', jax.random.normal, (self.hidden_units,))
        
        Wxc = self.param('Wxc', jax.random.normal, (self.input_dim, self.hidden_units))
        Whc = self.param('Whc', jax.random.normal, (self.hidden_units, self.hidden_units))
        bc = self.param('bc', jax.random.normal, (self.hidden_units,))
        
        def lstm_cell(h, c, x):
            I_t = jnp.sigmoid(jnp.dot(x, Wxi) + jnp.dot(h, Whi) + bi)
            F_t = jnp.sigmoid(jnp.dot(x, Wxf) + jnp.dot(h, Whf) + bf)
            O_t = jnp.sigmoid(jnp.dot(x, Wxo) + jnp.dot(h, Who) + bo)
            C_tilde = jnp.tanh(jnp.dot(x, Wxc) + jnp.dot(h, Whc) + bc)
            C = F_t * c + I_t * C_tilde
            H = O_t * jnp.tanh(C)
            return H, C
        
        _, all_hidden_states = nn.scan(lstm_cell, variable_axes={'params': 0}, split_rngs={'params': False})(inputs[:, 0, :], None, inputs)
        outputs = jnp.concatenate([h[None, ...] for h in all_hidden_states], axis=1)
        pred = nn.Dense(features=1)(outputs)
        
        return pred
    
# Define the LSTM Model
class LSTMModel(nn.Module):
    hidden_size: int
    
    @nn.compact
    def __call__(self, inputs):
        lstm = nn.LSTM(hidden_size=self.hidden_size, name='lstm')
        outputs, _ = lstm(inputs)
        pred = nn.Dense(features=1)(outputs[:, -1, :])  # Use the last output of the LSTM
        return pred
    
# Initialize the model, loss function, and optimizer
model_custom = CustomLSTMModel(input_dim=1, hidden_units=50)
model_inbuilt = LSTMModel(hidden_size=50)

def compute_loss(params, inputs, targets):
    preds, _ = model_custom.apply({'params': params}, inputs)
    return jnp.mean((preds[:, -1, :] - targets) ** 2)

# Training loop for the custom model
state = train_state.TrainState.create(apply_fn=model_custom.apply, params=model_custom.init(jax.random.PRNGKey(42), X_seq), tx=optimizer.adam(0.01))
epochs = 500
for epoch in range(epochs):
    state = state.apply_gradients(grads=compute_loss(state.params, X_seq, y_seq))
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {compute_loss(state.params, X_seq, y_seq).item():.4f}")

# Training loop for the inbuilt model
state = train_state.TrainState.create(apply_fn=model_inbuilt.apply, params=model_inbuilt.init(jax.random.PRNGKey(42), X_seq), tx=optimizer.adam(0.01))
epochs = 500
for epoch in range(epochs):
    state = state.apply_gradients(grads=compute_loss(state.params, X_seq, y_seq))
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {compute_loss(state.params, X_seq, y_seq).item():.4f}")

# Testing on new data
test_steps = 100  # Ensure this is greater than sequence_length
X_test = jnp.linspace(0, 5 * 3.14159, num=test_steps).reshape(-1, 1)
y_test = jnp.sin(X_test)

# Create test input sequences
X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

preds_custom = model_custom.apply({'params': state.params}, X_test_seq)
preds_inbuilt = model_inbuilt.apply({'params': state.params}, X_test_seq)

print(f"Predictions with Custom Model for new sequence: {jnp.squeeze(preds_custom).tolist()}")
print(f"Predictions with In-Built Model: {jnp.squeeze(preds_inbuilt).tolist()}")

# Plot the predictions
plt.figure()
plt.plot(y_test, label="Ground Truth")
plt.plot(jnp.squeeze(preds_custom), label="Custom LSTM")
plt.plot(jnp.squeeze(preds_inbuilt), label="In-Built LSTM")
plt.legend()
plt.show()