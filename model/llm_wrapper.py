import numpy as np
from model.transformer_block import transformer_block

class SimpleLLM:
    def __init__(self, vocab_size=1000, d_model=8, num_heads=4):
        self.d_model = d_model
        self.num_heads = num_heads
        # Real model would load trained weights, simulation for now
    
    def generate(self, prompt, max_length=10):
        """
        Simple generation: just pass through transformer block.
        Real case, there'd be tokenization, embedding, multiple blocks, etc.
        """
        # Simulate tokenization (random embeddings for now)
        # Later convert prompt to token IDs, then to embeddings
        seq_len = len(prompt.split())
        X = np.random.randn(seq_len, self.d_model)

        # Pass through transformer block
        output = transformer_block(X, self.num_heads)

        # Convert to text (simulated)
        return f"Generated response for: '{prompt}' (shape: {output.shape})"

