import jax
import jax.numpy as jnp
import jax.grad as grad
from jax import jit, vmap, grad, sin, cos
from jax.nn import Linear, sigmoid
from jax.scipy.stats import huber_loss
from jaxopt import minimize

# Generate synthetic data
rng = jnp.random.PRNGKey(42)
X = jnp.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jnp.random.normal(0, 1, (100, 1))  # Linear relationship with noise

class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta

    @jit
    def __call__(self, y_pred, y_true):
        error = jnp.abs(y_pred - y_true)
        loss = jax.ops.where(error <= self.delta, 0.5 * error**2, self.delta * (error - 0.5 * self.delta))
        return loss.mean()

class LinearRegressionModel:
    def __init__(self):
        self.W = jnp.ones((1, 1))
        self.b = jnp.zeros(1)

    @jit
    def __call__(self, X):
        return jax.nn.linear(X, self.W) + self.b

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
loss_fn = HuberLoss(delta=1.0)
opt_init, opt_update, opt_info = minimize(lambda params: -loss_fn(model(*X), y).item(), jnp.ones((2,)), method="BFGS")

# Training loop
for i in range(1000):
    with jax.grad(loss_fn):
        predictions = model(*X)
        loss = -loss_fn(predictions, y).item()

    grads = jax.grad(model.__call__)(X)[0]
    coefs, _ = opt_update(jax.grad(loss)(params=model.W, state=opt_init), grads)
    model.W, model.b = coefs[0], coefs[1]

    if (i + 1) % 100 == 0:
        print(f"Epoch [{i+1}/1000], Loss: {loss:.4f}")

# Display the learned parameters
w, b = model.W[0], model.b[0]
print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model(*X_test)
print(f"Predictions for {X_test}: {predictions}")