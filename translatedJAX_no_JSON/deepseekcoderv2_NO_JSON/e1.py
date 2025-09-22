import jax
from jax import numpy as jnp
import jaxlib.optix as optix

# Generate synthetic data
jax.random.seed(42)
key = jax.random.PRNGKey(42)
X = jax.random.uniform(key, (100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(key, (100, 1))  # Linear relationship with noise

# Define the Linear Regression Model
class LinearRegressionModel:
    def __init__(self):
        self.w = jnp.zeros((1,))
        self.b = jnp.zeros((1,))

    def __call__(self, x):
        return self.w * x + self.b

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()

def mean_squared_error(params, X, y):
    predictions = params['w'] * X + params['b']
    return jnp.mean((predictions - y) ** 2)

# Define the optimization step
@jax.jit
def update(params, opt_state, batch):
    loss, grads = jax.value_and_grad(mean_squared_error)(params, batch['X'], batch['y'])
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optix.apply_updates(params, updates)
    return params, opt_state, loss

# Training loop
epochs = 1000
batch_size = 32
learning_rate = 0.01

# Create batches for training
indices = jnp.arange(len(X))
batches = [{'X': X[indices[i:i + batch_size]], 'y': y[indices[i:i + batch_size]]} for i in range(0, len(X), batch_size)]

# Initialize optimizer state
params = {'w': jnp.zeros((1,)), 'b': jnp.zeros((1,))}
opt_state = optimizer.init(params)

for epoch in range(epochs):
    for batch in batches:
        params, opt_state, loss = update(params, opt_state, batch)
    
    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
print(f"Learned weight: {params['w'][0]:.4f}, Learned bias: {params['b'][0]:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model(X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")