import jax
from flax import linen as jnn
from jax import random
from torchvision import transforms
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

class SomeLayer(jnn.Module):
    features: int
    key: jax.random.PRNGKey

    def setup(self):
        self.dense = nn.Dense(self.features)

    def __call__(self, x):
        return jnn.relu(self.dense(x))  # Use a dense layer with relu activation

    def init(self, params, inputs, rngs=None):
        if rngs is None:
            key = jax.random.PRNGKey(0)
        else:
            self.key = rngs["rng"]
        subkey, self.key = random.split(self.key)  # Split the key for a new operation
        return params

    def apply(self, params, x, rngs=None):
        if rngs is None:
            key = jax.random.PRNGKey(0)
        else:
            self.key = rngs["rng"]
        subkey, self.key = random.split(self.key)  # Split the key for a new operation
        return self.dense(x)

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, key = random.split(key)  # Split key for randomness
    return random.normal(subkey, shape, dtype=dtype)

def main():
    key = jax.random.PRNGKey(0)
    input_tensor_shape = (10, 10)
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)

    layer = SomeLayer(features=5, key=key)
    params = layer.init({"params": key, "rng": key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={"rng": key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    output_np = np.array(output)  # Convert JAX array to NumPy
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert to PIL Image
    heatmap = transforms.Resize(image.size)(output_img)
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")  # Example title
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()