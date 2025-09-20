import jax
from jax import random, numpy as jnp
from flax import linen as jnn
from torchvision.transforms import Resize
from PIL import Image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class SomeLayer(jnn.Module):
    features: int

    def setup(self):
        self.dense = jnn.Dense(self.features)

    def __call__(self, x):
        return jnn.relu(self.dense(x))  # Use a dense layer with relu activation

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, key = random.split(key)  # Split key for randomness
    return random.normal(subkey, shape, dtype=dtype)  # Generate a tensor with specified dtype

def main():
    key = random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)

    layer = SomeLayer(features=5, key=key)
    params = layer.init({"params": key, "rng": key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={"rng": key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    output_np = np.array(output)  # Convert JAX array to NumPy
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert to PIL Image
    heatmap = Resize((image.size[0], image.size[1]), Image.NEAREST)(output_img)  # Use Image.NEAREST for resizing
    display_image = make_grid(images=[image, heatmap])  # Convert the PIL images to a tensor and display them using matplotlib
    plt.imshow(display_image.permute(1, 2, 0))
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()