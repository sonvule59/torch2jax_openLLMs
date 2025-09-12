import jax
import jax.numpy as jnp
from flax import linen as nn
import scipy.optimize

jax.random.seed(42)
X = jnp.ones((100, 1)) * 10  # 100 data points between 0 and 10
y = 2 * X + 3 + jax.random.normal(jax.random.PRNGKey(42), shape=(100, 1))  # Linear relationship with noise

class LearnedSiLUFunction:
    def __init__(self, slope):
        self.slope = slope

    @jax.jvp
    def __call__(self, inpt, v):
        def fwd(x):
            return self.slope * x * jnp.sigmoid(x)

        def bwd(grad_output):
            sigmoid_x = jnp.sigmoid(inpt)
            grad_input = grad_output * self.slope * (sigmoid_x + inpt * sigmoid_x * (1 - sigmoid_x))
            grad_slope = grad_output * inpt * sigmoid_x
            return [grad_input, None, grad_slope]
        return fwd, bwd

class LinearRegressionModel(nn.Module):
    def setup(self):
        self.slope = nn.Parameter(jnp.ones((1,)))

    def __call__(self, x):
        return LearnedSiLUFunction(self.slope)(x)

model = LinearRegressionModel()
loss = lambda pred, target: jnp.mean((pred - target) ** 2)
opt_func = scipy.optimize.minimize_scalar

# Initialize optimizer state
optimizer_state = opt_func(loss, model=model, args=(y,), method='L-BFGS-B', bounds=[(0.1, None)])

# Training loop
epochs = 1000
for epoch in range(epochs):
    predictions = model(X)
    loss_val = float(loss(predictions, y))

    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_val}")

    # Update optimizer state
    optimizer_state = opt_func(loss, model=model, args=(y,), **optimizer_state.kwargs)
    model.apply_parameters(**optimizer_state.x)

# Display the learned parameters
[w, b] = [p.value for p in model.slope.parameters()]
print(f"Learned weight: {w:.4f}, Learned bias: {b:.4f}")

# Testing on new data
X_test = jnp.array([[4.0], [7.0]])
with jax.grad():
    predictions = model(X_test)
print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")