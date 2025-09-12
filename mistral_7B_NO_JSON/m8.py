import jax
import jax.numpy as jnp
from jax import grad, vmap, random, jit, lax
from flax import linen as nn
import numpy as np
import jax.experimental.host_callback as hc
import jax.plot as jp

# Load MNIST dataset (This part is not supported in JAX)
# transform = ...
# train_dataset = ...
# test_dataset = ...
# train_loader = ...
# test_loader = ...

class Autoencoder(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv2D(1, 32, kernel_size=3, stride=1, padding="SAME")
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3, stride=1, padding="SAME")
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2D(2, 2)
        self.conv_transpose1 = nn.Conv2DTranspose(64, 32, kernel_size=3, stride=2, padding="SAME")
        self.relu3 = nn.ReLU()
        self.conv_transpose2 = nn.Conv2DTranspose(32, 1, kernel_size=3, stride=2, padding="SAME")

    def __call__(self, images):
        x = self.conv1(images)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv_transpose1(x)
        x = self.relu3(x)
        x = self.conv_transpose2(x)
        return jnp.clip(x, 0., 1.)

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optimizers.Adam(learning_rate=0.001)

def loss_fn(params):
    model = Autoencoder.from_params(params)
    def loss_grad(images, labels):
        reconstructed = model(images)
        return criterion(reconstructed, images)
    return vmap(loss_grad)(jax.random.uniform(shape=(64, 28, 28), dtype=jnp.float32))

optim_state = optimizer.initialize(model.init_params())

epochs = 10
for epoch in range(epochs):
    for images, _ in train_loader:
        with hc.stateful_bindings(optim_state):
            grads = jax.grad(loss_fn)(images)
        grads = jax.lax.psum(grads) / len(images)
        optim_state, new_params = optimizer.update(grads, optim_state)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss_fn(new_params).mean():.4f}")