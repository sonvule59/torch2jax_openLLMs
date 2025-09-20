import jax
import jax.numpy as jnp
import jax.grad as jgrad
import optax

# Generate synthetic data
rng = jax.random.PRNGKey(42)
X = jnp.array(jax.random.uniform(rng, (100, 1), minval=0, maxval=10))
y = 2 * X + 3 + jnp.random.normal(rng, (100, 1))  # Linear relationship with noise

# Define the Linear Regression Model
class LinearRegressionModel:
    def __init__(self):
        self.w = None
        self.b = None

    def init_params(self):
        self.w = optax.initializers.normal(rng, jnp.ones((1, 1)))
        self.b = optax.initializers.zeros()(rng)
        return self.w, self.b

    def predict(self, X):
        return jnp.dot(X, self.w) + self.b

# Initialize the model and optimizer
model = LinearRegressionModel()
model.init_params()
optimizer = optax.sgd(learning_rate=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    predictions = model.predict(X)
    loss = jnp.mean((predictions - y) ** 2)

    grads = jgrad(loss)(model.predict, (X,))
    _, vals = optimizer.apply_gradient(grads, model.init_params())
    model.w, model.b = vals

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Display the learned parameters
print(f"Learned weight: {model.w[0].item():.4f}, Learned bias: {model.b.item():.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
predictions = model.predict(X_test)
print(f"Predictions for {X_test}: {predictions}")