import numpy as np
import jax
import jax.numpy as jnp
import tenacity
from jax import random, grad, vmap, jit, lax
from functools import partial
from jax.experimental import optimizers
from tqdm import trange

# Helper function to convert PyTorch tensor-like object to JAX array
def pt_to_jax(pt):
    return np.array(pt).flatten() if isinstance(pt, list) else pt.numpy().flatten()

# Load CIFAR-10 dataset and convert into JAX arrays
key = random.PRNGKey(0)
transform = lax.vmap(lambda x: jax.nn.tanh(x) * (0.5 / jnp.sqrt(3.) + x))
train_dataset, _ = jax.datasets.cifar10(key, train=True, onehot=False)
test_dataset, _ = jax.datasets.cifar10(key, train=False, onehot=False)
train_loader, test_loader = data_loader(train_dataset, test_dataset, batch_size=64)

def data_loader(dataset, shuffle=True, batch_size=64):
    def batchify(data):
        return jax.random.shuffle(data, seed=0)[::batch_size]

    ds = jax.data.Dataset.from_generator(lambda i: data[i * batch_size : (i + 1) * batch_size], shuffle=shuffle)
    return ds.batch(batch_size)

# Define the CNN Model
def init_params():
    def rng_key, shape: generator -> jnp.ndarray:
        key, subkey = random.split(rng_key)
        return random.normal(subkey, shape)

    initial_weights = vmap(init_params)([((3, 32, 3, 3), key), ((32, 64, 3, 3), key), ((64 * 16 * 16, 128), key)], jax.random.PRNGKey(0))
    biases = vmap(init_params)([((32,), key), ((64,), key), ((10,), key)], jax.random.PRNGKey(0))
    return initial_weights, biases

@jit
def conv2d(x, w, b):
    return lax.conv2d(x, w, padding="SAME", strides=1) + b

@jit
def linear(x, w, b):
    return jnp.dot(w, x) + b

@jit
def model(params, x):
    w1, b1 = params[0]
    w2, b2 = params[1]
    w3, b3 = params[2]
    x = conv2d(x, w1, b1)
    x = lax.mean(x, axis=(1, 2))  # Average pooling
    x = conv2d(x, w2, b2)
    x = lax.mean(x, axis=1)       # Average pooling
    x = linear(x, w3, b3)
    return x

def loss_fn(params, x, y):
    logits = model(params, x)
    loss = jax.nn.softmax_cross_entropy(logits=logits, labels=y)
    return loss

# Initialize the model parameters and optimizer
key = random.PRNGKey(0)
params = init_params(key)
opt_init, opt_update, get_state = optimizers.adamax(learning_rate=0.001)
state = opt_init(params)

# Training loop
epochs = 10
for epoch in trange(epochs):
    for images, labels in train_loader:
        with tenacity.retrying(stop=tenacity.stop_after_attempt(5)):
            try:
                loss_grad = grad(loss_fn)(params, images, labels)
            except jax.errors.FunctionNotDefinedError:
                print("Gradient computation failed. Retrying...")
                continue
        state = opt_update(loss_grad, state)
        params = lax.psum(params, axis=0) / len(images)  # Update parameters with average gradient

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {lax.nanmean(pt_to_jax(loss_fn(params, train_loader[0][0], pt_to_jax(train_loader[0][1])))).item():.4f}")

# Evaluate on the test set
correct = 0
total = 0
for images, labels in test_loader:
    logits = model(params, images)
    predicted = jax.nn.logsoftmax(logits)[0].argmax(axis=1).reshape(-1)
    total += labels.shape[0]
    correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")