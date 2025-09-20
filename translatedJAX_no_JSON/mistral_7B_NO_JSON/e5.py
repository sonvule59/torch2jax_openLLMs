import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim

# Generate synthetic data
rng = jax.random.PRNGKey(42)
X = jnp.ones((100, 2)) * 10
key, subkey = jax.random.split(rng)
y = (X[:, 0] + X[:, 1] * 2).at[..., 0].add(jax.random.normal(subkey, (100,)))  # Non-linear relationship with noise

# Define the Deep Neural Network Model
class DNNModel(nn.Module):
    def setup(self):
        self.fc1 = nn.Dense(units=10)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Dense(units=1)

    def __call__(self, X):
        x = self.fc1(X)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = DNNModel()
criterion = nn.MSE()
optimizer = optim.sgd(learning_rate=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = model.apply(X, params=model.params)
    loss = criterion(predictions, y)

    # Backward pass and optimization
    grads = jax.grad(criterion)(predictions, y, params=model.params)
    optimizer.apply_gradient(grads, model.params)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testing on new data
X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
with jax.grad(computes_grad=False):
    predictions = model.apply(X_test, params=model.params)
    print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")