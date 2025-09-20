import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

class SomeLayer(nn.Module):
    features: int

    def setup(self):
        self.dense = nn.Dense(self.features)
        self.relu = jnn.relu

    def __call__(self, x):
        params = self.init({"params": random.PRNGKey(0), "rng": random.PRNGKey(0)}, x).named("params")  # Initialize the layer with parameters and x shape
        return self.relu(self.dense(x)).named("output")  # Use a dense layer with relu activation

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, key = random.split(key)  # Split key for randomness
    return random.normal(subkey, shape, dtype=dtype)

def main():
    key = random.PRNGKey(0)
    input_tensor_shape = (10, 10)
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)

    layer = SomeLayer(features=5, key=key)
    params = layer.init({"params": key, "rng": key}, input_tensor).named("params")  # Initialize the layer with parameters and x shape
    output = layer.apply(params, input_tensor, rngs={"rng": key}).named("output")

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = PIL.Image.fromarray(dummy_image_data)
    heatmap = transforms.Resize(image.size)(PIL.Image.fromarray(output.astype(np.uint8)))  # Convert JAX array to NumPy and then PIL Image, and apply the Resize transformation
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()