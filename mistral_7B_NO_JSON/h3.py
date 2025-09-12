import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim

class TransformerModel(nn.Module):
    input_dim: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    output_dim: int

    def setup(self):
        self.embedding = nn.Dense(self.embed_dim)
        self.transformer = nn.Transformer(num_heads=self.num_heads, hidden_dim=self.embed_dim, num_layers=self.num_layers)
        self.output_layer = nn.Dense(self.output_dim)

    def __call__(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = jnp.mean(x, axis=1)  # Pooling across the sequence
        return self.output_layer(x)

# Generate synthetic data
rng = jax.random.PRNGKey(42)
seq_length = 10
num_samples = 100
input_dim = 1
X = jnp.random.normal(rng, (num_samples, seq_length, input_dim))  # Random sequences
y = jnp.sum(X, axis=1)  # Target is the sum of each sequence

# Initialize the model, loss function, and optimizer
input_dim = 1
embed_dim = 16
num_heads = 2
num_layers = 2
ff_dim = 64
output_dim = 1

model = TransformerModel(input_dim, embed_dim, num_heads, num_layers, ff_dim, output_dim)
criterion = nn.MSE()
optimizer = optim.adam(optimizer_args={"learning_rate": 0.001})

# Training loop
epochs = 1000
for epoch in range(epochs):
    with jax.grad(criterion):
        predictions = model.apply(X)
        loss_value = criterion(predictions, y)

        grads = jax.grad(loss_value, model.params)

    optimizer.apply_gradient(grads=grads)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_value}")

# Testing on new data
X_test = jnp.random.normal(rng, (2, seq_length, input_dim))
with jax.no_grad():
    predictions = model.apply(X_test)
    print(f"Predictions for {X_test}: {predictions}")