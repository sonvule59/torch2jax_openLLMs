import jax
from jax import numpy as jnp
import flax.linen as nn
import optax

# Define a simple model using Flax Linen
class SimpleModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)

# Create and train the model
key = jax.random.PRNGKey(42)
model = SimpleModel()
params = model.init(key, jnp.ones((1, 1)))
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params)

# Training data
X = jnp.linspace(0, 2, 100).reshape(-1, 1)
y = 3 * X + 2 + jnp.random.normal(key, (100, 1), 0.1)
epochs = 100

# Training loop
def train_step(params, opt_state, X, y):
    def loss_fn(params):
        predictions = model.apply(params, X)
        return jnp.mean((predictions - y) ** 2)
    
    grads = jax.grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

for epoch in range(epochs):
    params, opt_state = train_step(params, opt_state, X, y)

# Save the model to a file named "model.pth"
import pickle
with open("model.pth", "wb") as f:
    pickle.dump(params, f)

# Load the model back from "model.pth"
with open("model.pth", "rb") as f:
    loaded_params = pickle.load(f)

# Verify the model works after loading
X_test = jnp.array([[0.5], [1.0], [1.5]])
predictions = model.apply(loaded_params, X_test)
print(f"Predictions after loading: {predictions}")