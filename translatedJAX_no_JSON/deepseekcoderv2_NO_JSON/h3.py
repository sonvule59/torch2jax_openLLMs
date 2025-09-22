import jax
from flax import linen as nn
import jax.numpy as np
import optax

# Define a Transformer Model using Flax
class TransformerModel(nn.Module):
    input_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.embed_dim)(x)
        x = nn.TransformerEncoder(
            layer=nn.TransformerEncoderLayer(
                features=self.embed_dim, 
                num_heads=self.num_heads, 
                dropout=0.1), 
            num_layers=self.num_layers)(x)
        x = np.mean(x, axis=1)  # Pooling across the sequence
        x = nn.Dense(self.output_dim)(x)
        return x

# Generate synthetic data using JAX/NumPy
jax.random.seed(0)
seq_length = 10
num_samples = 100
input_dim = 1
X = jax.random.normal(jax.random.PRNGKey(42), (num_samples, seq_length, input_dim))  # Random sequences
y = np.sum(X, axis=1)  # Target is the sum of each sequence

# Initialize the model, loss function, and optimizer using Optax
model = TransformerModel(input_dim=input_dim, embed_dim=16, num_heads=2, num_layers=2, ff_dim=64, output_dim=1)
loss_fn = nn.losses.mean_squared_error
optimizer = optax.adam(learning_rate=0.001)
state = optimizer.init(model.params)

# Training loop using JAX transformations
@jax.jit
def train_step(state, X, y):
    def loss_fn_(params):
        preds = model.apply({'params': params}, X)
        return np.mean((preds - y) ** 2)
    
    grads = jax.grad(loss_fn_)(state.params)
    updates, new_optimizer_state = optimizer.update(grads, state, model.params)
    new_params = optax.apply_updates(state.params, updates)
    return new_params, new_optimizer_state

epochs = 1000
batch_size = num_samples  # Full batch for simplicity
for epoch in range(epochs):
    params, state = train_step(state, X, y)
    
    if (epoch + 1) % 100 == 0:
        preds = model.apply({'params': params}, X)
        loss = np.mean((preds - y) ** 2)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Testing on new data
X_test = jax.random.normal(jax.random.PRNGKey(43), (2, seq_length, input_dim))
predictions = model.apply({'params': params}, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")