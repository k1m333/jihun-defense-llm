import numpy as np

def softmax(x, axis=-1):
    """Stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    """Single-head self-attention (reused from before)."""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    d_k = W_Q.shape[1]
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V
    return output, attention_weights

def multi_head_attention(X, num_heads, d_model):
    """
    Multi-head self-attention.
    X : (n, d_model) – input embeddings
    num_heads : number of attention heads
    d_model : embedding dimension (must be divisible by num_heads)
    Returns:
        output : (n, d_model) – concatenated and projected output
    """
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_k = d_model // num_heads
    n = X.shape[0]
    
    # Initialize weight matrices for all heads
    # Random weights for demonstration
    np.random.seed(42)
    W_Q = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    W_K = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    W_V = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
    W_O = np.random.randn(num_heads * d_k, d_model)  # Output projection
    
    # Run each head
    head_outputs = []
    for h in range(num_heads):
        out, _ = self_attention(X, W_Q[h], W_K[h], W_V[h])  # (n, d_k)
        head_outputs.append(out)
    
    # Concatenate along last dimension
    concat = np.concatenate(head_outputs, axis=-1)  # (n, num_heads * d_k)
    
    # Project back to d_model
    output = concat @ W_O  # (n, d_model)
    
    return output

if __name__ == "__main__":
    # Test with small example
    np.random.seed(42)
    n, d_model = 3, 8
    num_heads = 2
    X = np.random.randn(n, d_model)
    
    out = multi_head_attention(X, num_heads, d_model)
    print("Input shape:", X.shape)
    print("Output shape:", out.shape)  # Should be (3, 8)
