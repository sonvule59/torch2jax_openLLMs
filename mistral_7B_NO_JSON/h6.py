import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim
from jax.experimental import quantization

class LanguageModel(nn.Module):
    vocab_size: int
    embed_size: int
    hidden_size: int
    num_layers: int

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTMCell(self.embed_size, self.hidden_size)
        self.fc = nn.Dense(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax()

    def __call__(self,carry, inputs):
        carry[0], output = self.lstm(self.embedding(inputs), carry[0])
        return (carry, self.fc(output)[..., jnp.newaxis, :])

    def init(self, rng, Carry=nn.initializers.zeros):
        return Carry(rng, (self.num_layers, self.hidden_size))

# Create synthetic training data
rng = jax.random.PRNGKey(42)
vocab_size = 50
seq_length = 10
batch_size = 32
X_train = jax.random.randint(rng, (batch_size, seq_length), 0, vocab_size-1)  # Random integer input
y_train = jax.random.randint(rng, (batch_size,))  # Random target words

# Initialize the model, loss function, and optimizer
embed_size = 64
hidden_size = 128
num_layers = 2
model = LanguageModel(vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)

criterion = nn.LogSoftmaxCrossEntropy()
optimizer = optim.Adam(step_size=0.001, betas=(0.9, 0.999))

# Training loop
for epoch in range(5):
    carry = model.init_carry(rng)
    loss = jax.vmap(lambda x, y: jax.tree_multimap(lambda a, b: criterion(a, y), model(carry, x), y_train))()
    grads = jax.grad(loss)(X_train)
    optimizer.apply_gradients(grads=grads, params=model.parameters())

    # Log progress every epoch
    print(f"Epoch [{epoch + 1}/{5}] - Loss: {loss.item():.4f}")

# Quantization: Apply dynamic quantization to the language model
quantizer = quantization.create_qlinear_qrng_policy('QINT8', min=-127, max=128)
quantized_model = quantization.apply_qrng(model, policy=quantizer)