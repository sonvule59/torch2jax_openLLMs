import jax
from jax import random, jnn, numpy as jnp
from flax import linen as jflax
from torchvision.transforms import Resize
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

class SomeLayer(jflax.linen.Module):
    features: int = None

    def setup(self):
        self.dense = jnn.Dense(self.features)

    def __call__(self, x):
        return jnn.relu(self.dense(x))  # Use a dense layer with relu activation

    @staticmethod
    def generate_random_tensor(shape, dtype=jnp.float32, key=None):
        if key is None:
            raise ValueError("PRNG key must be provided")  # Error handling for missing key
        subkey, key = random.split(key)  # Split key for randomness
        return random.normal(subkey, shape, dtype=dtype)  # Generate a tensor with specified dtype

def main():
    key = random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = SomeLayer.generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)

    layer = SomeLayer()  # Initialize the module without passing the PRNGKey explicitly
    params = layer.init({"params": key, "rng": key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={"rng": key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    heatmap = Resize(image.size)(Image.fromarray(output.numpy().astype(np.uint8)))

    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")  # Example title
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()