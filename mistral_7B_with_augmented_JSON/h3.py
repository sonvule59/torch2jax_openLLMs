import jax
from flax import linen as jnn
from torchvision.transforms import Resize
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

class SomeLayer(jnn.Module):
    features: int
    key: jax.random.PRNGKey

    def setup(self):
        self.dense = nn.Dense(self.features)

    def __call__(self, x, rngs=None):
        return jnn.relu(self.dense(x))

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
    params = layer.init({"params": key, "rng": key}, input_tensor)
    output = layer.apply(params, input_tensor, rngs={"rng": key})

    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)

    output_np = np.array(output)
    output_img = ImageOps.grayscale(Image.fromarray(output_np))
    heatmap = Resize((100, 100))(output_img)

    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()