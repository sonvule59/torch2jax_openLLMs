import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from PIL import Image
from torchvision import transforms

class SomeLayer(nn.Module):
    features: int
    
    def setup(self):
        self.key = self.make_rng('rng')  # Create a separate RNG key for operations
        
    def __call__(self, x):
        subkey = self.make_rng('rng')  # Split the key for randomness in each call
        return nn.relu(nn.Dense(self.features)(x))  # Use a dense layer with relu activation

def generate_random_tensor(shape, dtype=jnp.float32, key=None):
    if key is None:
        raise ValueError("PRNG key must be provided")
    subkey, key = jax.random.split(key)  # Split the key for randomness
    return jax.random.normal(subkey, shape, dtype=dtype)

def main():
    key = jax.random.PRNGKey(0)  # Initialize a PRNGKey
    input_tensor_shape = (10, 10)  # Define the shape of the input tensor
    input_tensor = generate_random_tensor(input_tensor_shape, dtype=jnp.float32, key=key)
    
    layer = SomeLayer(features=5, key=key)  # Initialize the layer with parameters
    params = layer.init({'params': key, 'rng': key}, input_tensor)  # Initialize the layer with parameters
    output = layer.apply(params, input_tensor, rngs={'rng': key})  # Process the input tensor with initialized parameters
    
    dummy_image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(dummy_image_data)
    output_np = np.array(output)  # Convert JAX array to NumPy array
    output_img = Image.fromarray(output_np.astype(np.uint8))  # Convert to PIL Image
    heatmap = transforms.Resize(image.size)(output_img)  # Apply the transformation
    
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title("Predicted Class: Example Class")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()