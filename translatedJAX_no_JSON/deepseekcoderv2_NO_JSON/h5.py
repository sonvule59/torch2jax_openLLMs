import jax
import jaxlib.xla_client as xla
from jax import numpy as jnp
import optax

# Define the Encoder
class Encoder(jax.experimental.nn.Module):
    def apply(self, input_dim, embed_dim, hidden_dim, num_layers):
        self.embedding = jax.vmap(lambda x: jax.nn.one_hot(x, input_dim))(embed_dim)
        lstm = jax.experimental.LSTMCell(hidden_dim, (num_layers, embed_dim))
        return lambda x: lstm(x).apply(None, x)

# Define the Decoder with Attention
class Decoder(jax.experimental.nn.Module):
    def apply(self, output_dim, embed_dim, hidden_dim, num_layers, src_seq_length):
        self.embedding = jax.vmap(lambda x: jax.nn.one_hot(x, output_dim))(embed_dim)
        attention = jax.experimental.Linear(hidden_dim + embed_dim, src_seq_length)
        attention_combine = jax.experimental.Linear(hidden_dim + embed_dim, embed_dim)
        lstm = jax.experimental.LSTMCell(hidden_dim, (num_layers, embed_dim))
        fc_out = jax.experimental.Linear(hidden_dim, output_dim)
        
        def forward(x, encoder_outputs, hidden, cell):
            x = jnp.expand_dims(x, 1)
            embedded = self.embedding(x)
            
            # Attention mechanism
            attention_weights = jax.nn.softmax(attention(jnp.concatenate((embedded.squeeze(1), hidden[-1]), axis=1)), axis=1)
            context_vector = jnp.sum(attention_weights.unsqueeze(1) * encoder_outputs, axis=2)
            
            # Combine context and embedded input
            combined = jnp.concatenate((embedded.squeeze(1), context_vector.squeeze(1)), axis=1)
            combined = jax.nn.tanh(attention_combine(combined)).unsqueeze(1)
            
            # LSTM and output
            lstm_out, (hidden, cell) = lstm(combined, hidden, cell)
            output = fc_out(lstm_out.squeeze(1))
            return output, hidden, cell
        
        return forward

# Define synthetic training data
jax.random.PRNGKey(42)  # Set seed for reproducibility
src_vocab_size = 20
tgt_vocab_size = 20
src_seq_length = 10
tgt_seq_length = 12
batch_size = 16

src_data = jax.random.randint(jax.random.PRNGKey(42), (batch_size, src_seq_length), minval=0, maxval=src_vocab_size)
tgt_data = jax.random.randint(jax.random.PRNGKey(42), (batch_size, tgt_seq_length), minval=0, maxval=tgt_vocab_size)

# Initialize models, loss function, and optimizer
input_dim = src_vocab_size
output_dim = tgt_vocab_size
embed_dim = 32
hidden_dim = 64
num_layers = 2

encoder = Encoder(input_dim, embed_dim, hidden_dim, num_layers)
decoder = Decoder(output_dim, embed_dim, hidden_dim, num_layers, src_seq_length)

def loss_fn(params, batch):
    src_data, tgt_data = batch
    encoder_outputs, (hidden, cell) = encoder.apply(params['encoder'], None, src_data)
    
    def decode_step(carry, x):
        output, hidden, cell = decoder.apply(params['decoder'], carry, None, x)
        return (output, hidden, cell), output
    
    initial_carry = (jax.random.PRNGKey(42), hidden, cell)
    _, outputs = jax.lax.scan(decode_step, initial_carry, tgt_data[:, :-1])
    
    loss = optax.softmax_cross_entropy_with_integer_labels(outputs, tgt_data[:, 1:]).mean()
    return loss

optimizer = optax.adam(learning_rate=0.001)
state = optimizer.init(jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params))

# Training loop
epochs = 100
for epoch in range(epochs):
    keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
    batches = [(src_data[i], tgt_data[i]) for i in range(batch_size)]
    
    def train_step(state, key, batch):
        params = state.optimizer_state
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss, new_params, new_opt_state
    
    losses, params, opt_states = jax.lax.scan(train_step, (state, keys), batches)
    
    # Log progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {losses.mean():.4f}")

# Test the sequence-to-sequence model with new input
test_input = jax.random.randint(jax.random.PRNGKey(42), (1, src_seq_length), minval=0, maxval=src_vocab_size)
encoder_outputs, (hidden, cell) = encoder.apply(params['encoder'], None, test_input)
initial_carry = (jax.random.PRNGKey(42), hidden, cell)

def decode_step(carry, x):
    output, hidden, cell = decoder.apply(params['decoder'], carry, None, x)
    return (output, hidden, cell), output

_, outputs = jax.lax.scan(decode_step, initial_carry, tgt_data[:, :-1])
predicted = jnp.argmax(outputs[-1], axis=1)
print(f"Input: {test_input.tolist()}, Output: {predicted.tolist()}")