import jax
from jax import grad, jit, random, vmap
from jax.experimental import mobility
from jax.nn import Linear
from jax.optimizers import adam
from jax.scipy.special import sqrt
from jax.experimental.mobility import MixedPrecisionTrainingConfig, MixedPrecisionGradientTape

# Define a simple model
def simple_model(params):
    w = params[0]
    return Linear(w)(jax.ops.reshape(jax.random.normal(jax.random.PRNGKey(0), (1, 10)), (10, -1))).T

# Generate synthetic data
X = random.normal(jax.random.PRNGKey(1), (1000, 10))
y = random.normal(jax.random.PRNGKey(2), (1000, 1))

# Initialize model parameters and optimizer
init_params = random.normal(jax.random.PRNGKey(3), (1, 10))
rng, key = jax.random.split(jax.random.PRNGKey(4))
opt_state = adam.create(learning_rate=0.001, rngs=(rng, key))

# Set up mixed precision training with 8-bit floats (JAX uses bfloat16 by default)
config = MixedPrecisionTrainingConfig(
    master_precision="bfloat16",
    gradient_precision="float32",
    use_offloading=True,
    offload_threshold=0.85,
    max_num_mixed_ops=100000,
)
tape = MixedPrecisionGradientTape(config=config)

# Training loop
for epoch in range(5):
    for i, (x, y) in enumerate(zip(X.split(32), y.split(32))):
        with tape.record():
            params = init_params
            loss, grads = jax.value_and_grad(simple_model)(params)(x)
        grads = grad(loss)
        optimizer_update, opt_state = adam.update_rule(grads, opt_state)
        params = mobility.update_parameters(init_params, optimizer_update)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

# Test the model on new data
X_test = random.normal(jax.random.PRNGKey(5), (5, 10))
params = mobility.apply_updates(init_params, mobility.apply_optimizer_state(opt_state))
predictions = simple_model(params)(jax.ops.reshape(X_test, (5, -1)))
print("Predictions:", predictions)