import jax
import jax.numpy as jnp
from jax import grad, vmap
import flax.linen as nn
import optax

# Define the Generator
class Generator(nn.Module):
    latent_dim: int
    data_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.data_dim)(x)
        x = nn.tanh(x)
        return x

# Define the Discriminator
class Discriminator(nn.Module):
    input_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dense(128)(x)
        x = nn.leaky_relu(x, 0.2)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x

# Generate synthetic data for training
key = jax.random.PRNGKey(42)
real_data = (jax.random.uniform(key, (100, 1)) * 2 - 1).astype(jnp.float32)  # 100 samples in the range [-1, 1]
latent_dim = 10
data_dim = 1

# Initialize models
G = Generator(latent_dim=latent_dim, data_dim=data_dim)
D = Discriminator(input_dim=data_dim)

# Define loss function
def bce_loss(logits, labels):
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

# Define optimizers
tx_G = optax.adam(learning_rate=0.001)
tx_D = optax.adam(learning_rate=0.001)

opt_G = tx_G.init(G.params)
opt_D = tx_D.init(D.params)

# Training loop
@jax.jit
def train_step_G(opt_state, latent_samples):
    def loss_fn(params):
        fake_data = G.apply({'params': params}, latent_samples).astype(jnp.float32)
        return bce_loss(D.apply({'params': params}, fake_data), jnp.ones((real_data.shape[0], 1)))
    grads = grad(loss_fn)(opt_state.target)
    updates, opt_state = tx_G.update(grads, opt_state)
    return opt_state, G.params, updates

@jax.jit
def train_step_D(opt_state, real_data, latent_samples):
    def loss_fn(params):
        fake_data = G.apply({'params': params}, latent_samples).astype(jnp.float32)
        real_labels = jnp.ones((real_data.shape[0], 1))
        fake_labels = jnp.zeros((real_data.shape[0], 1))
        real_loss = bce_loss(D.apply({'params': params}, real_data), real_labels)
        fake_loss = bce_loss(D.apply({'params': params}, fake_data), fake_labels)
        return real_loss + fake_loss
    grads = grad(loss_fn)(opt_state.target)
    updates, opt_state = tx_D.update(grads, opt_state)
    return opt_state, D.params, updates

epochs = 1000
for epoch in range(epochs):
    # Train Discriminator
    key, subkey = jax.random.split(key)
    latent_samples = jax.random.normal(subkey, (real_data.shape[0], latent_dim))
    opt_D, D_params, _ = train_step_D(opt_D, real_data, latent_samples)

    # Train Generator
    key, subkey = jax.random.split(key)
    latent_samples = jax.random.normal(subkey, (real_data.shape[0], latent_dim))
    opt_G, G_params, _ = train_step_G(opt_G, latent_samples)

    # Log progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

# Generate new samples with the trained Generator
key, subkey = jax.random.split(key)
latent_samples = jax.random.normal(subkey, (5, latent_dim))
generated_data = G.apply({'params': G_params}, latent_samples).astype(jnp.float32)
print(f"Generated data: {generated_data.tolist()}")