import jax
import jaxlib.xla_client as xc
from jax import numpy as np
import matplotlib.pyplot as plt
import optax

# Generate synthetic data
rng = jax.random.PRNGKey(42)
X = jax.random.uniform(rng, (100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(rng, (100, 1))  # Linear relationship with noise

# Define the Linear Regression Model within a CustomActivationModel class
class CustomActivationModel:
    def __init__(self):
        self.linear = lambda x: np.tanh(x) + x

    def __call__(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
model = CustomActivationModel()

def compute_loss(params, inputs, targets):
    predictions = model(inputs)
    return np.mean((predictions - targets) ** 2)

optimizer = optax.sgd(learning_rate=0.01)

# Initialize parameters for the model
params = {
    'linear': {'kernel': jax.random.uniform(jax.random.PRNGKey(43), (1, 1)), 'bias': jax.random.uniform(jax.random.PRNGKey(44), (1,))}
}

# Training loop
epochs = 1000
loss_history = []
for epoch in range(epochs):
    # Compute gradients
    grads = jax.grad(compute_loss)(params, X, y)
    
    # Update parameters using the optimizer
    updates, opt_state = optimizer.update(grads['linear'], params['linear'], optax.State(count=0))
    new_params = optax.apply_updates(params['linear'], updates)
    params['linear'] = new_params
    
    # Compute loss for logging
    loss = compute_loss(params, X, y)
    loss_history.append(loss)
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
w = params['linear']['kernel']
b = params['linear']['bias']
print(f"Learned weight: {w[0, 0]:.4f}, Learned bias: {b[0]:.4f}")

# Plot the model fit to the train data
plt.figure(figsize=(4, 4))
plt.scatter(X, y, label='Training Data')
plt.plot(X, w[0, 0] * X + b[0], 'r', label='Model Fit')
plt.legend()
plt.show()

# Testing on new data
X_test = np.array([[4.0], [7.0]])
predictions = model(X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")