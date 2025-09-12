import jax
from flax import linen as nn
import jax.numpy as jnp

class PositionalEncoding(nn.Module):
    def setup(self):
        self.position = self.param("positions", nn.initializers.trunc_normal(), (1, src_seq_length, embed_dim))

    def __call__(self, x):
        position_embedding = jnp.sin(self.position[:, 0::2]) + jnp.cos(self.position[:, 1::2])
        return x + position_embedding

class Encoder(nn.Module):
    def setup(self):
        self.embed = nn.Dense(embed_dim)
        self.pos_enc = PositionalEncoding()
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)

    def __call__(self, x):
        x = self.pos_enc(x)
        x = self.embed(x)
        state = jnp.zeros((batch_size, num_layers, hidden_dim))
        for layer in range(num_layers):
            state = self.lstm(x, state[layer])
        return state[-1]

class Decoder(nn.Module):
    def setup(self):
        self.embed = nn.Dense(embed_dim)
        self.attention = Attention(hidden_dim, src_seq_length)
        self.combine = nn.Dense(embed_dim)
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.fc_out = nn.Dense(output_dim)

    def __call__(self, x, encoder_outputs, state):
        x = self.embed(x)

        attention_weights = self.attention(x, encoder_outputs, state[-1])
        context_vector = jnp.sum(encoder_outputs * attention_weights, axis=0)

        combined = jnp.concatenate([x, context_vector], axis=-1)
        combined = jnp.tanh(self.combine(combined))
        state = self.lstm(combined, state[-1])
        return self.fc_out(state), state

class Attention(nn.Module):
    def setup(self, hidden_dim, src_seq_length):
        self.query = nn.Dense(hidden_dim)
        self.key = nn.Dense(hidden_dim)
        self.value = nn.Dense(hidden_dim)
        self.attention_combine = nn.Dense(hidden_dim)

    def __call__(self, q, k, v):
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        attention_scores = jnp.matmul(query, jnp.transpose(key)) / jnp.sqrt(hidden_dim)
        attention_weights = jnp.softmax(attention_scores, axis=-1)
        context_vector = jnp.matmul(attention_weights, value)
        return context_vector

# Define synthetic training data
jnp.random.seed(42)
src_vocab_size = 20
tgt_vocab_size = 20
src_seq_length = 10
tgt_seq_length = 12
batch_size = 16

src_data = jnp.random.randint(0, src_vocab_size, (batch_size, src_seq_length))
tgt_data = jnp.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_length))

# Initialize models and optimizer
input_dim = src_vocab_size
output_dim = tgt_vocab_size
embed_dim = 32
hidden_dim = 64
num_layers = 2

encoder = Encoder()
decoder = Decoder()
optimizer = optax.adam(learning_rate=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    encoder_outputs, state = encoder.init(src_data)
    loss = 0
    decoder_input = jnp.zeros((batch_size, 1), dtype=jnp.int32)  # Start token

    for t in range(tgt_seq_length):
        output, state = decoder(decoder_input, encoder_outputs, state)
        loss += jax.tree_map(lambda x: -jax.log(x), crossentropy(output, tgt_data[:, t]))
        decoder_input = tgt_data[:, t]  # Teacher forcing

    grads = jax.grad(loss)(encoder.init, encoder.apply, decoder.init, decoder.apply, state)
    optimizer = optimizer.update(grads)
    states = (encoder.init, decoder.init)

    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.mean():.4f}")

# Test the sequence-to-sequence model with new input
test_input = jnp.random.randint(0, src_vocab_size, (1, src_seq_length))
with jax.random.PRNGKey(42):
    state = encoder.init(test_input)
    decoder_input = jnp.zeros((1, 1), dtype=jnp.int32)  # Start token
    output_sequence = []

    for _ in range(tgt_seq_length):
        output, state = decoder(decoder_input, encoder_outputs, state)
        predicted = jnp.argmax(output, axis=-1).squeeze()
        output_sequence.append(predicted.item())
        decoder_input = predicted

    print(f"Input: {test_input}, Output: {output_sequence}")