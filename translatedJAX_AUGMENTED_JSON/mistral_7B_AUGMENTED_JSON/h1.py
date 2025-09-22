import jax
from jax import numpy as jnp
from flax import linen as nn
from jaxopt import optimize, value_and_grad

rng = jax.random.PRNGKey(42)
X = jnp.random.uniform(rng, (100, 1), minval=0.0, maxval=10.0) * 10.0
y = 2.0 * X + 3.0 + jnp.random.normal(rng, (100, 1))

class LearnedSiLUFunction:
    def __call__(self, x, slope):
        sigmoid_x = jax.nn.sigmoid(x)
        return slope * x * sigmoid_x

class LinearRegressionModel(nn.Module):
    def setup(self):
        self.slope = nn.Parameter(value=jnp.ones((1,)), name="slope")

    def __call__(self, x):
        return LearnedSiLUFunction()(x, self.slope)

model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optimize.sgd(step_size=0.01)

for epoch in range(1000):
    predictions = model.apply({"params": model.parameters}, X)
    loss = criterion(predictions, y)
    grads = value_and_grad(loss)(model.apply, (X,), {"params": model.parameters})["grads"]
    optimizer.apply_gradient(grads, model.parameters)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

[w, b] = model.parameters()
print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

X_test = jnp.array([[4.0], [7.0]])
with jax.grad():
    predictions = model(X_test)
print(f"Predictions for {X_test}: {predictions}")