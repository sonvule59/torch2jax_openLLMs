from flax import linen as jnn
import jax
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class SomeLayer(jnn.Module):
    features: int
    key: jnp.ndarray

    def setup(self):
        self.dense = nn.Dense(self.features)

    def __call__(self, x):
        return jnn.relu(self.dense(x))  # Use a dense layer with relu activation

    def init(self, params_rng, input_shape=None):
        key, subkey = jax.random.split(params_rng)
        self.params = self.dense.initialize(subkey, input_shape)
        self.key = key
        return self.params

    def apply(self, params, x, rngs):
        key = rngs['rng']
        subkey, key = jax.random.split(key)  # Split the key for a new operation
        return self.dense.apply(params, x, subkey)

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")  # Error handling for missing key
    return jax.random.normal(key, shape, dtype=dtype)

def main():
    key = jax.random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)

    layer = SomeLayer(features=5, key=key)
    params = layer.init(key, input_tensor.shape[1:])  # Initialize the layer with parameters
    output = layer.apply(params, jnp.ones((1,) + input_tensor_shape), rngs={'rng': key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    output_np = np.array(output)  # Convert JAX array to NumPy
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert to PIL Image
    heatmap = transforms.Resize(image.size)(output_img)  # Apply the transformation to the heatmap
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")  # Example title
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()