import jax
from flax import linen as jnn
from PIL import Image, ImageOps
from torchvision.transforms import Resize
from matplotlib.pyplot as plt
import numpy as np

class SomeLayer(jnn.Module):
    features: int

    def setup(self):
        self.dense = jnn.Dense(self.features)

    def __call__(self, x):
        return self.dense(x)

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, key = jax.random.split(key)
    return jax.random.normal(subkey, shape, dtype=dtype)

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
    output_img = ImageOps.grayscale(Image.fromarray(output_np))  # Convert to grayscale PIL image for Grad-CAM

    heatmap = Resize((image.size[0], image.size[1]), Image.BICUBIC)(output_img)  # Apply the transformation on the PIL image

    plt.imshow(image, cmap='gray')
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()