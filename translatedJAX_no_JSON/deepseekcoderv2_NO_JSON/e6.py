import jax
from jax import numpy as jnp
from jax.tree_util import Partial
from jaxlib.xla_client import local_devices
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from torch.utils.tensorboard import SummaryWriter

# Generate synthetic data
np.random.seed(42)
X = (np.random.rand(100, 1) * 10).astype(np.float32)  # 100 data points between 0 and 10
y = (3 * X + 5 + np.random.randn(100, 1)).astype(np.float32)  # Linear relationship with noise

# Define a simple Linear Regression Model using Flax
class LinearRegressionModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=1)(x)
        return x

# Initialize the model, loss function, and optimizer
model = LinearRegressionModel()
params = model.init(jax.random.PRNGKey(42), jnp.ones((1, 1)))['params']
optimizer = optax.sgd(learning_rate=0.01)
tx = optimizer.create(params)

# Define loss function
def compute_loss(params, x, y):
    predictions = LinearRegressionModel().apply({'params': params}, x)
    return jnp.mean((predictions - y) ** 2)

# Training loop using JAX's custom training step
@jax.jit
def train_step(state, x, y):
    loss, grads = jax.value_and_grad(compute_loss)(state.params, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss

# Training loop with TensorBoard logging
epochs = 100
writer = SummaryWriter(log_dir="runs/linear_regression")

for epoch in range(epochs):
    state, loss = train_step(tx, X, y)
    
    # Log loss to TensorBoard
    writer.add_scalar("Loss/train", loss, epoch)

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

# Close the TensorBoard writer
writer.close()