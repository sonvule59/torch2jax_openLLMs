import jax
from jax import numpy as jnp, grad, jit
from jax.optimizers import gradient_descent
import matplotlib.pyplot as plt

# Generate synthetic data
jax.random.seed(42)
X = jnp.ones((100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(jax.random.PRNGKey(42), (100, 1))  # Linear relationship with noise

class CustomActivationModel:
    def __init__(self):
        self.w = None
        self.b = None

    def custom_activation(self, x):
        return jnp.tanh(x) + x

    def forward(self, X):
        if self.w is None or self.b is None:
            self.w = jnp.ones((1, 1))
            self.b = jnp.zeros((1, 1))
        return self.custom_activation(jax.lax.dot(self.w, X) + self.b)

# Initialize the model, loss function, and optimizer
model = CustomActivationModel()
criterion = jax.vmap(lambda pred, y_true: (pred - y_true)**2).mean
opt_init, opt_update, opt_state = gradient_descent(jit(grad(criterion)))

# Training loop
epochs = 1000
for epoch in range(epochs):
    with jax.grad(criterion):
        predictions = model.forward(X)
        loss, _ = criterion(predictions, y).backward()

    opt_state = opt_update(opt_state, (-1 * X.flatten(),))  # Pass the data in flattened form

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss}")

# Display the learned parameters
model.w, model.b = jax.lax.conj(model.w), jax.lax.conj(model.b)  # For comparison with PyTorch, JAX uses complex numbers for gradients
print(f"Learned weight: {model.w[0].imag:.4f}, Learned bias: {model.b[0].imag:.4f}")

# Plot the model fit to the train data
plt.figure(figsize=(4, 4))
plt.scatter(X.flatten(), y)
plt.plot(X.flatten(), (model.w[0]*X.flatten() + model.b[0]), 'r')
plt.show()

# Testing on new data
X_test = jnp.array([4.0, 7.0])
with jax.no_grad():
    predictions = model.forward(X_test)
    print(f"Predictions for {X_test}: {predictions}")