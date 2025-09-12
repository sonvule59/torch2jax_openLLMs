import jax
from flax import linprog
from flax import linen as nn
from jax import random, grad, vmap, jit
from jax.experimental import optimizers

class SimpleModel(nn.Module):
    def setup(self):
        self.fc = nn.Dense(1)

    def __call__(self, x):
        return self.fc(x)

rng_key = random.PRNGKey(42)

# Initialize the model and optimizer
model = SimpleModel()
optimizer = optimizers.sgd(learning_rate=0.01)

def loss(params, x, y):
    predictions = model.apply({"fc": params}, x)
    return jax.lax.mean((predictions - y) ** 2)

# Training loop
X = random.normal(rng_key, (100, 1))
y = 3 * X + 2 + random.normal(rng_key, (100, 1)) * 0.1
epochs = 100
for epoch in range(epochs):
    grads, _ = jax.grad(loss)(model.parameters), optimizer.update(model.parameters, loss(model.parameters, X, y))

# Save the model to a file named "model.jda" using JAX's checkpoint API
import jax.numpy as jnp
from jax import tree_multimap, pmap
from jax.experimental import checkpoint

def save_checkpoint(model):
    def _save(params):
        return {"fc": params["fc"]}

    x = pmap(_save)(model.parameters)
    checkpoint.checkpoint(x, "model.jda")

# Load the model back from "model.jda" using JAX's checkpoint API
def load_checkpoint():
    def _load(params):
        return tree_multimap(nn.initializers.truncated_normal, params)

    x = checkpoint.checkpoint("model.jda")
    loaded_model = SimpleModel()
    loaded_model.initialize(tree_unflatten(SimpleModel, x))

# Verify the model works after loading
X_test = jnp.array([[0.5], [1.0], [1.5]])
loaded_model = load_checkpoint()
predictions = loaded_model(X_test)
print(f"Predictions after loading: {predictions}")