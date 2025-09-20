import numpy as np
import jax
from jax import grad, jit, random, vmap
from jax.nn import Linear, sigmoid
from jax.experimental.optimizers import optimize
from jax.experimental.metrics import mean_squared_error

# Generate synthetic data
rng = random.PRNGKey(42)
X = vmap(lambda x: np.array(x)*10 + 5)(random.normal(rng, (100, 1)))  # 100 data points between 0 and 10 with noise added
y = X * 3 + random.normal(rng, (100, 1))  # Linear relationship with noise

# Define a simple Linear Regression Model
linear_model = jit(lambda params: Linear(features_axis=1)(params[0]) @ params[1] + params[2])

# Initialize the model parameters and optimizer
init_rng, opt_rng = random.split(rng)
params = linear_model.initialize(np.array([[0.]]), np.array([np.eye(1), np.zeros((1, 1))]), X)
opt_state = optimize(jax.gradient(mean_squared_error), args=(y,))

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    with opt_rng:
        grads, loss = jax.value_and_grad(mean_squared_error)(params[0], y)
        grads, loss = jax.lax.psum(grads), jax.lax.psum(loss)  # average over the data points

    opt_state = optimize.update(opt_state, grads, params)
    params = opt_state.params

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")