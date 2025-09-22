import jax
from jax import numpy as jnp
from jax import random
import flax.linen as nn
from flax.optim import SGD

# Generate synthetic data
key = random.PRNGKey(42)
X = (random.uniform(key, (100, 1)) * 10).astype(jnp.float32)  # 100 data points between 0 and 10
y = 2 * X + 3 + random.normal(key, (100, 1)).astype(jnp.float32)  # Linear relationship with noise

# Define the Huber Loss function
def huber_loss(params, inputs, targets, delta):
    predictions = nn.Dense(features=1).apply({'params': params}, inputs)
    error = jnp.abs(predictions - targets)
    loss = jnp.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return jnp.mean(loss)

# Define the Linear Regression Model using Flax
class LinearRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1)(x)
        return x

# Initialize the model parameters
model = LinearRegressionModel()
params = model.init(key, jnp.ones((1, 1)))['params']

# Define the optimizer
optimizer = SGD(learning_rate=0.01)
state = optimizer.create(params)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Compute gradients and update parameters
    def loss_fn(params):
        return huber_loss(params, X, y, delta=1.0)
    
    grads = jax.grad(loss_fn)(state.params)
    state = optimizer.apply_gradient(grads, state)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        predictions = model.apply({'params': state.params}, X)
        loss = huber_loss(state.params, X, y, delta=1.0)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
w = state.params['Dense_0']['kernel']
b = state.params['Dense_0']['bias']
print(f"Learned weight: {float(w[0]):.4f}, Learned bias: {float(b[0]):.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model.apply({'params': state.params}, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")