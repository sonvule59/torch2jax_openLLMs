import jax
from flax import linen as jnn
from jax.experimental import jax2numpy
from PIL import Image
from torchvision.transforms import Resize
import numpy as np
import matplotlib.pyplot as plt

class SomeLayer(jnn.Module):
  def __init__(self, features=5, key=None):
    super().__init__()
    self._dense = jnn.Dense(features)

  def init(self, params, inputs=None, rngs=None):
    # Initialize the layer with parameters and randomness
    key, subkey = jax.random.split(rngs['rng']) if rngs else jax.random.split(jax.random.PRNGKey(0))
    self.params = params['params']
    return params

  def __call__(self, x):
    # Pass the input through a dense layer with ReLU activation
    return jnn.relu(self._dense(x))

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

  # Generate a synthetic image in memory
  dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
  image = Image.fromarray(dummy_image_data)

  # Convert JAX array to NumPy for compatibility with torchvision.transforms.Resize
  output_np = jax2numpy.as_numpy(output)

  # Apply the transformation and convert the result back to a PIL image
  heatmap = Resize(image.size)(Image.fromarray(output_np.astype(np.uint8)))

  # Display the image with the Grad-CAM heatmap
  plt.imshow(image)
  heatmap.putalpha(0.5)
  plt.imshow(heatmap, cmap='jet')
  plt.title("Predicted Class: Example Class")
  plt.axis('off')
  plt.show()

if __name__ == "__main__":
  main()