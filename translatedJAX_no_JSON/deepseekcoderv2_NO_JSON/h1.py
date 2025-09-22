import jax
from jax import numpy as jnp
import flax.linen as nn
import optax

# Generate synthetic data
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(key, (100, 1))  # Linear relationship with noise

class LearnedSiLUFunction(nn.Module):
    slope: float

    def setup(self):
        self.slope_jax = self.param('slope', nn.initializers.ones, (), self.slope)

    def __call__(self, x):
        return self.slope_jax * x * jax.nn.sigmoid(x)

# Define the Linear Regression Model
def linear_regression_model():
    return LearnedSiLUFunction(slope=1.0)

# Initialize the model and optimizer
model = linear_regression_model()
optimizer = optax.sgd(learning_rate=0.01)
state = optimizer.init(model.params)

# Define loss function
def mse_loss(params, state, x, y):
    preds = model.apply({'params': params}, x)
    return jnp.mean((preds - y) ** 2)

# Training loop
epochs = 1000
for epoch in range(epochs):
    grad_fn = jax.value_and_grad(mse_loss, has_aux=True)
    (loss,), grads = grad_fn(model.params, state, X, y)
    updates, new_state = optimizer.update(grads, state, model.params)
    params = optax.apply_updates(model.params, updates)
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
print(f"Learned weight: {model.params['slope']:.4f}, Learned bias: {0:.4f}")  # Assuming no bias term in SiLU function for simplicity

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model.apply({'params': params}, X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")