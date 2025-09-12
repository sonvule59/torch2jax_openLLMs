import jax
from jax import random, grad, vmap, lax
import numpy as np
import pandas_jax as pdj

np.random.seed(42)
X = jax.random.uniform(key=random.PRNGKey(42), shape=(100, 1)) * 10
y = 2 * X + 3 + jax.random.normal(key=random.PRNGKey(42), shape=(100, 1))
data = jax.concatenate((X, y), axis=-1)
df = pdj.DataFrame(data, columns=['X', 'y'])
df.to_csv('data.csv', index=False)

import jax.numpy as jnp
from jax import jit, grad, vmap, random, lax
from flax import linprog

def create_opt_fn(params):
    def opt_fn(grads):
        updates, _ = linprog.linear_update(params, grads, 0.01)
        return [parameters.modify(updates[i]) for i, parameters in enumerate(params)]
    return opt_fn

def linear_regression_model():
    parameters = linprog.Parameters(jnp.zeros((2,)))
    return linprog.LinearRegressionModel(parameters)

@jit
def loss(params, X, y):
    model = linear_regression_model()
    predictions = vmap(model.predict)(X, params)
    return lax.mean((predictions - y)**2)

@jit
def grad_loss(params, X, y):
    model = linear_regression_model()
    grad_fn = jax.grad(loss)(params)
    grads = vmap(grad_fn)(X, y)
    return grads

params = linear_regression_model().init(jax.random.PRNGKey(42), jnp.ones((2,)))
opt_fn = create_opt_fn(params)

# Manual batching function
def get_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

X, y = jax.device_put(data[:100], 'cpu'), jax.device_put(data[100:], 'cpu')
batches_x, batches_y = get_batches(X, 32), get_batches(y, 32)

epochs = 1000
for epoch in range(epochs):
    for i, (batch_X, batch_y) in enumerate(zip(batches_x, batches_y)):
        params = opt_fn(grad_loss(params, jax.device_put(batch_X, 'cpu'), jax.device_put(batch_y, 'cpu')))

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss(params, X, y).item():.4f}")

# Display the learned parameters
[w, b] = params[0].value(), params[1].value()
print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

X_test = jax.random.normal(jax.random.PRNGKey(42), shape=(2, 1))
with jax.no_grad():
    predictions = vmap(linear_regression_model().predict)(X_test, params)
    print(f"Predictions for {X_test.tolist()}: {predictions[0].item():.4f}, {predictions[1].item():.4f}")