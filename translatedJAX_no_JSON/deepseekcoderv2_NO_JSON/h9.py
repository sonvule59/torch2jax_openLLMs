import jax
import jaxlib
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers
from jax.tree_util import tree_flatten, tree_unflatten
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define a simple model using Flax (a JAX library for neural networks)
import flax.linen as nn
import optax

# Generate synthetic data
X = jnp.random.normal(0, 1, size=(1000, 10))
y = jnp.random.normal(0, 1, size=(1000, 1))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
model = nn.Dense(features=1)(nn.Dense(features=10)(nn.Input(10)))
params = model.init(jax.random.PRNGKey(0), X)
loss_fn = lambda params, x, y: jnp.mean((model.apply(params, x) - y) ** 2)
optimizer = optax.adam(learning_rate=0.001)
state = optimizer.init(params)

# Training loop
epochs = 5
for epoch in range(epochs):
    for inputs, labels in dataloader:
        inputs, labels = jnp.array(inputs), jnp.array(labels)
        
        def train_step(state, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)
            updates, new_state = optimizer.update(grads, state, params=state.params)
            new_params = optax.apply_updates(state.params, updates)
            return loss, new_state, new_params
        
        loss, state, params = train_step(state, inputs, labels)
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Test the model on new data
X_test = jnp.random.normal(0, 1, size=(5, 10))
predictions = model.apply(params, X_test)
print("Predictions:", predictions)