import jax
import jaxlib
from jax import numpy as jnp
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 2)) * 10
y = (X[:, 0] + X[:, 1] * 2).reshape(-1, 1) + jax.random.normal(key, (100, 1))

# Define the Deep Neural Network Model
class DNNModel:
    def __init__(self):
        self.params = {
            'fc1': {'weights': jnp.zeros((2, 10)), 'bias': jnp.zeros(10)},
            'fc2': {'weights': jnp.zeros((10, 1)), 'bias': jnp.zeros(1)},
        }

    def __call__(self, x):
        x = x @ self.params['fc1']['weights'] + self.params['fc1']['bias']
        x = jax.nn.relu(x)
        x = x @ self.params['fc2']['weights'] + self.params['fc2']['bias']
        return x

# Initialize the model, loss function, and optimizer
model = DNNModel()
loss_fn = lambda params, x, y: jnp.mean((model(x, params) - y) ** 2)
optimizer = optax.adam(learning_rate=0.01)
state = optimizer.init(model.params)

# Training loop
epochs = 1000
for epoch in range(epochs):
    grads = jax.grad(loss_fn)(model.params, X, y)
    updates, state = optimizer.update(grads, state)
    model.params = optax.apply_updates(model.params, updates)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        loss = loss_fn(model.params, X, y)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Testing on new data
X_test = jnp.array([[4.0, 3.0], [7.0, 8.0]])
predictions = model(X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")